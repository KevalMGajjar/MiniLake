//! Binder: sqlparser AST -> [`LogicalPlan`].
//!
//! Responsibilities:
//! * resolve table names against the [`Catalog`];
//! * qualify every column reference with its relation alias;
//! * assemble joins (explicit `JOIN ... ON` and comma joins with equality
//!   predicates in `WHERE`) into a left-deep tree of equi-joins;
//! * extract aggregates and build the `Aggregate` node;
//! * resolve ORDER BY aliases / positions;
//! * light type coercion of literals (e.g. `'1995-03-15'` vs a DATE column).

use minilake_core::date::{add_months, parse_date};
use minilake_core::{DataType, MiniLakeError, Result, ScalarValue};
use minilake_exec::expr::BinaryOp;
use minilake_storage::Catalog;
use sqlparser::ast;

use crate::logical::{
    aggregate_type, AggFunc, ColumnRef, Expr, LogicalPlan, LogicalSchema, QField, SortExpr,
};
use crate::optimizer::constant_folding::fold_expr;

fn unsupported<T>(what: impl std::fmt::Display) -> Result<T> {
    Err(MiniLakeError::Unsupported(what.to_string()))
}

/// Binds statements against a catalog.
pub struct Binder<'a> {
    catalog: &'a Catalog,
}

impl<'a> Binder<'a> {
    /// New binder.
    pub fn new(catalog: &'a Catalog) -> Self {
        Binder { catalog }
    }

    /// Bind a parsed statement (only queries are supported).
    pub fn bind_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Query(q) => self.bind_query(q),
            other => unsupported(format!("statement: {other}")),
        }
    }

    /// Bind a query.
    pub fn bind_query(&self, q: &ast::Query) -> Result<LogicalPlan> {
        if q.with.is_some() {
            return unsupported("WITH / CTEs");
        }
        let select = match q.body.as_ref() {
            ast::SetExpr::Select(s) => s,
            ast::SetExpr::Query(inner) => return self.bind_query(inner),
            other => return unsupported(format!("query body: {other}")),
        };
        let order_by: Vec<&ast::OrderByExpr> = match &q.order_by {
            None => vec![],
            Some(ob) => match &ob.kind {
                ast::OrderByKind::Expressions(e) => e.iter().collect(),
                ast::OrderByKind::All(_) => return unsupported("ORDER BY ALL"),
            },
        };
        let (limit, offset) = match &q.limit_clause {
            None => (None, 0),
            Some(ast::LimitClause::LimitOffset { limit, offset, .. }) => (
                limit.as_ref().map(const_usize).transpose()?,
                offset
                    .as_ref()
                    .map(|o| const_usize(&o.value))
                    .transpose()?
                    .unwrap_or(0),
            ),
            Some(ast::LimitClause::OffsetCommaLimit { offset, limit }) => {
                (Some(const_usize(limit)?), const_usize(offset)?)
            }
        };
        self.bind_select(select, &order_by, limit, offset)
    }

    fn bind_select(
        &self,
        select: &ast::Select,
        order_by: &[&ast::OrderByExpr],
        limit: Option<usize>,
        offset: usize,
    ) -> Result<LogicalPlan> {
        if select.distinct.is_some() {
            return unsupported("SELECT DISTINCT");
        }
        // ---- FROM + WHERE --------------------------------------------------
        let (mut plan, from_schema) = self.bind_from(select)?;
        let input_schema = plan.schema();

        // ---- SELECT items --------------------------------------------------
        struct Item {
            expr: Expr,
            name: String,
            relation: Option<String>,
            has_alias: bool,
        }
        let mut items: Vec<Item> = Vec::new();
        for si in &select.projection {
            match si {
                ast::SelectItem::Wildcard(_) => {
                    for f in &input_schema.fields {
                        items.push(Item {
                            expr: Expr::Column(ColumnRef {
                                relation: f.relation.clone(),
                                name: f.name.clone(),
                            }),
                            name: f.name.clone(),
                            relation: f.relation.clone(),
                            has_alias: false,
                        });
                    }
                }
                ast::SelectItem::UnnamedExpr(e) => {
                    let expr = self.bind_expr(e, &from_schema)?;
                    let (name, relation) = match &expr {
                        Expr::Column(c) => (c.name.clone(), c.relation.clone()),
                        other => (other.to_string(), None),
                    };
                    items.push(Item {
                        expr,
                        name,
                        relation,
                        has_alias: false,
                    });
                }
                ast::SelectItem::ExprWithAlias { expr, alias } => {
                    items.push(Item {
                        expr: self.bind_expr(expr, &from_schema)?,
                        name: alias.value.to_ascii_lowercase(),
                        relation: None,
                        has_alias: true,
                    });
                }
                other => return unsupported(format!("select item {other}")),
            }
        }

        // ---- GROUP BY / HAVING / ORDER BY (bound, not yet rewritten) --------
        let mut group_exprs = Vec::new();
        match &select.group_by {
            ast::GroupByExpr::Expressions(exprs, mods) => {
                if !mods.is_empty() {
                    return unsupported("GROUP BY modifiers");
                }
                for e in exprs {
                    // GROUP BY 1 / GROUP BY alias
                    let bound = if let Some(pos) = positional(e) {
                        items
                            .get(pos)
                            .map(|i| i.expr.clone())
                            .ok_or_else(|| MiniLakeError::Plan(format!("GROUP BY {} out of range", pos + 1)))?
                    } else if let Some(i) = alias_ref(e).and_then(|a| {
                        items.iter().find(|i| i.has_alias && i.name == a)
                    }) {
                        i.expr.clone()
                    } else {
                        self.bind_expr(e, &from_schema)?
                    };
                    group_exprs.push(bound);
                }
            }
            ast::GroupByExpr::All(_) => return unsupported("GROUP BY ALL"),
        }
        let having = match &select.having {
            Some(h) => Some(self.bind_expr(h, &from_schema)?),
            None => None,
        };

        // ORDER BY: either an output column (alias / position) or an expression.
        enum SortTarget {
            Output(usize),
            Expr(Expr),
        }
        let mut sort_targets: Vec<(SortTarget, bool, bool)> = Vec::new();
        for ob in order_by {
            let asc = !matches!(ob.options.sort, Some(ast::OrderBySort::Desc));
            // SQL default: NULLS LAST for ASC, NULLS FIRST for DESC (DuckDB: NULLS LAST always)
            let nulls_first = ob.options.nulls_first.unwrap_or(false);
            let target = if let Some(pos) = positional(&ob.expr) {
                if pos >= items.len() {
                    return Err(MiniLakeError::Plan(format!("ORDER BY {} out of range", pos + 1)));
                }
                SortTarget::Output(pos)
            } else if let Some(i) = alias_ref(&ob.expr).and_then(|a| {
                items.iter().position(|i| i.has_alias && i.name == a)
            }) {
                SortTarget::Output(i)
            } else {
                SortTarget::Expr(self.bind_expr(&ob.expr, &from_schema)?)
            };
            sort_targets.push((target, asc, nulls_first));
        }

        // ---- Aggregation ---------------------------------------------------
        let needs_agg = !group_exprs.is_empty()
            || items.iter().any(|i| i.expr.contains_aggregate())
            || having.as_ref().is_some_and(|h| h.contains_aggregate());
        let mut having = having;
        if needs_agg {
            let mut aggregates: Vec<Expr> = Vec::new();
            let mut collect = |e: &Expr| {
                e.visit(&mut |x| {
                    if matches!(x, Expr::Aggregate { .. }) && !aggregates.contains(x) {
                        aggregates.push(x.clone());
                    }
                })
            };
            for i in &items {
                collect(&i.expr);
            }
            if let Some(h) = &having {
                collect(h);
            }
            for (t, _, _) in &sort_targets {
                if let SortTarget::Expr(e) = t {
                    collect(e);
                }
            }
            // Output schema: group keys then aggregates.
            let mut fields = Vec::new();
            let mut group_refs = Vec::new();
            for (gi, g) in group_exprs.iter().enumerate() {
                let (relation, name) = match g {
                    Expr::Column(c) => (c.relation.clone(), c.name.clone()),
                    _ => (None, format!("__group{gi}")),
                };
                fields.push(QField {
                    relation: relation.clone(),
                    name: name.clone(),
                    data_type: g.data_type(&input_schema)?,
                    nullable: true,
                });
                group_refs.push((g.clone(), ColumnRef { relation, name }));
            }
            let mut agg_refs = Vec::new();
            for (ai, a) in aggregates.iter().enumerate() {
                if let Expr::Aggregate {
                    func,
                    arg: Some(arg),
                } = a
                {
                    if arg.contains_aggregate() {
                        return Err(MiniLakeError::Plan("nested aggregates".into()));
                    }
                    aggregate_type(*func, Some(arg.data_type(&input_schema)?))?;
                }
                let name = format!("__agg{ai}");
                fields.push(QField {
                    relation: None,
                    name: name.clone(),
                    data_type: a.data_type(&input_schema)?,
                    nullable: true,
                });
                agg_refs.push((
                    a.clone(),
                    ColumnRef {
                        relation: None,
                        name,
                    },
                ));
            }
            let agg_schema = LogicalSchema { fields };
            plan = LogicalPlan::Aggregate {
                input: Box::new(plan),
                group_exprs: group_exprs.clone(),
                aggregates,
                schema: agg_schema.clone(),
            };
            let rewrite = |e: Expr| -> Result<Expr> {
                let out = e.rewrite_top_down(&mut |x| {
                    if let Some((_, r)) = group_refs.iter().find(|(g, _)| g == x) {
                        return Some(Expr::Column(r.clone()));
                    }
                    if let Some((_, r)) = agg_refs.iter().find(|(a, _)| a == x) {
                        return Some(Expr::Column(r.clone()));
                    }
                    None
                })?;
                let mut cols = Vec::new();
                out.columns(&mut cols);
                if let Some(bad) = cols.iter().find(|c| agg_schema.resolve(c).is_err()) {
                    return Err(MiniLakeError::Plan(format!(
                        "column '{bad}' must appear in GROUP BY or be used in an aggregate"
                    )));
                }
                Ok(out)
            };
            for i in &mut items {
                i.expr = rewrite(std::mem::replace(&mut i.expr, Expr::Literal(ScalarValue::Null)))?;
            }
            if let Some(h) = having.take() {
                having = Some(rewrite(h)?);
            }
            for (t, _, _) in &mut sort_targets {
                if let SortTarget::Expr(e) = t {
                    *e = rewrite(std::mem::replace(e, Expr::Literal(ScalarValue::Null)))?;
                }
            }
        }
        if let Some(h) = having {
            let schema = plan.schema();
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: coerce(h, &schema)?,
            };
        }

        // ---- Projection (+ hidden sort columns) ------------------------------
        let proj_input = plan.schema();
        let mut exprs = Vec::new();
        let mut fields = Vec::new();
        let mut used_names: Vec<String> = Vec::new();
        for i in &items {
            let mut name = i.name.clone();
            let mut k = 1;
            while used_names.contains(&name) {
                name = format!("{}_{k}", i.name);
                k += 1;
            }
            used_names.push(name.clone());
            let expr = coerce(i.expr.clone(), &proj_input)?;
            fields.push(QField {
                relation: i.relation.clone(),
                name,
                data_type: expr.data_type(&proj_input)?,
                nullable: true,
            });
            exprs.push(expr);
        }
        let visible = exprs.len();
        let mut sort_keys = Vec::new();
        for (t, asc, nulls_first) in sort_targets {
            let idx = match t {
                SortTarget::Output(i) => i,
                SortTarget::Expr(e) => {
                    let e = coerce(e, &proj_input)?;
                    match exprs.iter().position(|x| *x == e) {
                        Some(i) => i,
                        None => {
                            let name = format!("__sort{}", exprs.len());
                            fields.push(QField {
                                relation: None,
                                name,
                                data_type: e.data_type(&proj_input)?,
                                nullable: true,
                            });
                            exprs.push(e);
                            exprs.len() - 1
                        }
                    }
                }
            };
            let f = &fields[idx];
            sort_keys.push(SortExpr {
                expr: Expr::Column(ColumnRef {
                    relation: f.relation.clone(),
                    name: f.name.clone(),
                }),
                asc,
                nulls_first,
            });
        }
        let proj_schema = LogicalSchema { fields };
        plan = LogicalPlan::Projection {
            input: Box::new(plan),
            exprs,
            schema: proj_schema.clone(),
        };
        if !sort_keys.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                keys: sort_keys,
            };
        }
        if limit.is_some() || offset > 0 {
            plan = LogicalPlan::Limit {
                input: Box::new(plan),
                limit: limit.unwrap_or(usize::MAX),
                offset,
            };
        }
        if proj_schema.fields.len() > visible {
            let fields: Vec<QField> = proj_schema.fields[..visible].to_vec();
            plan = LogicalPlan::Projection {
                input: Box::new(plan),
                exprs: fields
                    .iter()
                    .map(|f| {
                        Expr::Column(ColumnRef {
                            relation: f.relation.clone(),
                            name: f.name.clone(),
                        })
                    })
                    .collect(),
                schema: LogicalSchema { fields },
            };
        }
        Ok(plan)
    }

    /// FROM + WHERE: returns the plan and the combined schema of all relations
    /// (used to qualify column references).
    fn bind_from(&self, select: &ast::Select) -> Result<(LogicalPlan, LogicalSchema)> {
        let mut relations: Vec<LogicalPlan> = Vec::new();
        let mut on_conditions: Vec<&ast::Expr> = Vec::new();
        for twj in &select.from {
            relations.push(self.bind_table(&twj.relation)?);
            for j in &twj.joins {
                match &j.join_operator {
                    ast::JoinOperator::Inner(c) | ast::JoinOperator::Join(c) => match c {
                        ast::JoinConstraint::On(e) => on_conditions.push(e),
                        ast::JoinConstraint::None => {}
                        other => return unsupported(format!("join constraint {other:?}")),
                    },
                    ast::JoinOperator::CrossJoin(_) => {}
                    other => return unsupported(format!("join type {other:?}")),
                }
                relations.push(self.bind_table(&j.relation)?);
            }
        }
        if relations.is_empty() {
            return unsupported("SELECT without FROM");
        }
        let combined = relations
            .iter()
            .skip(1)
            .fold(relations[0].schema(), |acc, r| acc.join(&r.schema()));

        let mut conjuncts = Vec::new();
        for e in on_conditions.into_iter().chain(select.selection.as_ref()) {
            self.bind_expr(e, &combined)?.split_conjunction(&mut conjuncts);
        }

        // Greedy left-deep join assembly in FROM order: repeatedly join the
        // first remaining relation that is connected to the current tree by
        // at least one equality predicate.
        let mut current = relations.remove(0);
        while !relations.is_empty() {
            let cs = current.schema();
            let pick = relations.iter().position(|r| {
                let rs = r.schema();
                conjuncts.iter().any(|c| equi_pair(c, &cs, &rs).is_some())
            });
            let Some(i) = pick else {
                return unsupported(
                    "cross join (every table must be connected by an equality predicate)",
                );
            };
            let right = relations.remove(i);
            let rs = right.schema();
            let mut on = Vec::new();
            conjuncts.retain(|c| match equi_pair(c, &cs, &rs) {
                Some(pair) => {
                    on.push(pair);
                    false
                }
                None => true,
            });
            let schema = cs.join(&rs);
            current = LogicalPlan::Join {
                left: Box::new(current),
                right: Box::new(right),
                on,
                schema,
            };
        }
        if let Some(pred) = Expr::conjunction(conjuncts) {
            let schema = current.schema();
            current = LogicalPlan::Filter {
                input: Box::new(current),
                predicate: coerce(pred, &schema)?,
            };
        }
        Ok((current, combined))
    }

    fn bind_table(&self, factor: &ast::TableFactor) -> Result<LogicalPlan> {
        let ast::TableFactor::Table { name, alias, .. } = factor else {
            return unsupported(format!("FROM item {factor}"));
        };
        let table_name = name
            .0
            .last()
            .map(|p| p.to_string())
            .unwrap_or_default()
            .trim_matches('"')
            .to_ascii_lowercase();
        let table = self.catalog.table(&table_name)?;
        let alias = alias
            .as_ref()
            .map(|a| a.name.value.to_ascii_lowercase())
            .unwrap_or_else(|| table_name.clone());
        let schema = LogicalSchema {
            fields: table
                .schema()
                .fields
                .iter()
                .map(|f| QField {
                    relation: Some(alias.clone()),
                    name: f.name.to_ascii_lowercase(),
                    data_type: f.data_type,
                    nullable: f.nullable,
                })
                .collect(),
        };
        Ok(LogicalPlan::Scan {
            table,
            alias,
            projection: None,
            filters: Vec::new(),
            schema,
        })
    }

    /// Convert an AST expression; column references are qualified against `schema`.
    pub fn bind_expr(&self, e: &ast::Expr, schema: &LogicalSchema) -> Result<Expr> {
        use ast::Expr as A;
        Ok(match e {
            A::Identifier(id) => qualify(
                ColumnRef {
                    relation: None,
                    name: id.value.to_ascii_lowercase(),
                },
                schema,
            )?,
            A::CompoundIdentifier(ids) if ids.len() == 2 => qualify(
                ColumnRef {
                    relation: Some(ids[0].value.to_ascii_lowercase()),
                    name: ids[1].value.to_ascii_lowercase(),
                },
                schema,
            )?,
            A::Value(v) => Expr::Literal(bind_value(&v.value)?),
            A::TypedString(ts) => {
                let ty = ts.data_type.to_string().to_ascii_uppercase();
                let text = match &ts.value.value {
                    ast::Value::SingleQuotedString(s) => s.clone(),
                    other => other.to_string(),
                };
                if ty.starts_with("DATE") {
                    Expr::Literal(ScalarValue::Date(parse_date(&text).ok_or_else(|| {
                        MiniLakeError::Plan(format!("invalid date literal '{text}'"))
                    })?))
                } else {
                    Expr::Cast {
                        expr: Box::new(Expr::Literal(ScalarValue::Utf8(text))),
                        to: map_type(&ts.data_type)?,
                    }
                }
            }
            A::Nested(inner) => self.bind_expr(inner, schema)?,
            A::UnaryOp { op, expr } => {
                let inner = self.bind_expr(expr, schema)?;
                match op {
                    ast::UnaryOperator::Not => Expr::Not(Box::new(inner)),
                    ast::UnaryOperator::Minus => Expr::Negate(Box::new(inner)),
                    ast::UnaryOperator::Plus => inner,
                    other => return unsupported(format!("unary operator {other}")),
                }
            }
            A::BinaryOp { left, op, right } => {
                // DATE +/- INTERVAL
                if let A::Interval(iv) = right.as_ref() {
                    let sign = match op {
                        ast::BinaryOperator::Plus => 1,
                        ast::BinaryOperator::Minus => -1,
                        _ => return unsupported("interval arithmetic other than +/-"),
                    };
                    let base = self.bind_expr(left, schema)?;
                    return apply_interval(base, iv, sign);
                }
                let l = self.bind_expr(left, schema)?;
                let r = self.bind_expr(right, schema)?;
                let op = match op {
                    ast::BinaryOperator::Plus => BinaryOp::Add,
                    ast::BinaryOperator::Minus => BinaryOp::Sub,
                    ast::BinaryOperator::Multiply => BinaryOp::Mul,
                    ast::BinaryOperator::Divide => BinaryOp::Div,
                    ast::BinaryOperator::Modulo => BinaryOp::Mod,
                    ast::BinaryOperator::Eq => BinaryOp::Eq,
                    ast::BinaryOperator::NotEq => BinaryOp::NotEq,
                    ast::BinaryOperator::Lt => BinaryOp::Lt,
                    ast::BinaryOperator::LtEq => BinaryOp::LtEq,
                    ast::BinaryOperator::Gt => BinaryOp::Gt,
                    ast::BinaryOperator::GtEq => BinaryOp::GtEq,
                    ast::BinaryOperator::And => BinaryOp::And,
                    ast::BinaryOperator::Or => BinaryOp::Or,
                    other => return unsupported(format!("operator {other}")),
                };
                Expr::binary(l, op, r)
            }
            A::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let x = self.bind_expr(expr, schema)?;
                let between = Expr::binary(
                    Expr::binary(x.clone(), BinaryOp::GtEq, self.bind_expr(low, schema)?),
                    BinaryOp::And,
                    Expr::binary(x, BinaryOp::LtEq, self.bind_expr(high, schema)?),
                );
                if *negated {
                    Expr::Not(Box::new(between))
                } else {
                    between
                }
            }
            A::InList {
                expr,
                list,
                negated,
            } => Expr::InList {
                expr: Box::new(self.bind_expr(expr, schema)?),
                list: list
                    .iter()
                    .map(|x| self.bind_expr(x, schema))
                    .collect::<Result<_>>()?,
                negated: *negated,
            },
            A::Like {
                negated,
                any,
                expr,
                pattern,
                escape_char,
            } => {
                if *any || escape_char.is_some() {
                    return unsupported("LIKE ANY / ESCAPE");
                }
                let pattern = match self.bind_expr(pattern, schema)? {
                    Expr::Literal(ScalarValue::Utf8(s)) => s,
                    _ => return unsupported("LIKE with a non-constant pattern"),
                };
                Expr::Like {
                    expr: Box::new(self.bind_expr(expr, schema)?),
                    pattern,
                    negated: *negated,
                }
            }
            A::IsNull(e) => Expr::IsNull {
                expr: Box::new(self.bind_expr(e, schema)?),
                negated: false,
            },
            A::IsNotNull(e) => Expr::IsNull {
                expr: Box::new(self.bind_expr(e, schema)?),
                negated: true,
            },
            A::Case {
                operand,
                conditions,
                else_result,
                ..
            } => {
                let operand = match operand {
                    Some(o) => Some(self.bind_expr(o, schema)?),
                    None => None,
                };
                let mut branches = Vec::new();
                for cw in conditions {
                    let mut cond = self.bind_expr(&cw.condition, schema)?;
                    if let Some(o) = &operand {
                        cond = Expr::binary(o.clone(), BinaryOp::Eq, cond);
                    }
                    branches.push((cond, self.bind_expr(&cw.result, schema)?));
                }
                Expr::Case {
                    branches,
                    else_expr: match else_result {
                        Some(e) => Some(Box::new(self.bind_expr(e, schema)?)),
                        None => None,
                    },
                }
            }
            A::Cast {
                expr, data_type, ..
            } => Expr::Cast {
                expr: Box::new(self.bind_expr(expr, schema)?),
                to: map_type(data_type)?,
            },
            A::Function(f) => self.bind_function(f, schema)?,
            other => return unsupported(format!("expression {other}")),
        })
    }

    fn bind_function(&self, f: &ast::Function, schema: &LogicalSchema) -> Result<Expr> {
        let name = f.name.to_string().to_ascii_lowercase();
        let ast::FunctionArguments::List(list) = &f.args else {
            return unsupported(format!("function call {f}"));
        };
        if list.duplicate_treatment.is_some()
            && !matches!(list.duplicate_treatment, Some(ast::DuplicateTreatment::All))
        {
            return unsupported(format!("{name}(DISTINCT ...)"));
        }
        if f.over.is_some() || f.filter.is_some() {
            return unsupported("window functions / FILTER");
        }
        let func = match name.as_str() {
            "count" => AggFunc::Count,
            "sum" => AggFunc::Sum,
            "avg" => AggFunc::Avg,
            "min" => AggFunc::Min,
            "max" => AggFunc::Max,
            other => return unsupported(format!("function {other}")),
        };
        if list.args.len() != 1 {
            return Err(MiniLakeError::Plan(format!("{name} takes one argument")));
        }
        match &list.args[0] {
            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Wildcard) if func == AggFunc::Count => {
                Ok(Expr::Aggregate {
                    func: AggFunc::CountStar,
                    arg: None,
                })
            }
            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e)) => Ok(Expr::Aggregate {
                func,
                arg: Some(Box::new(self.bind_expr(e, schema)?)),
            }),
            other => unsupported(format!("function argument {other}")),
        }
    }
}

/// Qualify a column reference with its relation using `schema`.
fn qualify(c: ColumnRef, schema: &LogicalSchema) -> Result<Expr> {
    let i = schema.resolve(&c)?;
    let f = schema.field(i);
    Ok(Expr::Column(ColumnRef {
        relation: f.relation.clone(),
        name: f.name.clone(),
    }))
}

/// If `c` is `a = b` with `a` from `left` and `b` from `right` (or swapped),
/// return `(left_expr, right_expr)`.
fn equi_pair(c: &Expr, left: &LogicalSchema, right: &LogicalSchema) -> Option<(Expr, Expr)> {
    let Expr::Binary {
        op: BinaryOp::Eq,
        left: a,
        right: b,
    } = c
    else {
        return None;
    };
    let has_cols = |e: &Expr| {
        let mut v = Vec::new();
        e.columns(&mut v);
        !v.is_empty()
    };
    if !has_cols(a) || !has_cols(b) {
        return None;
    }
    if left.can_resolve(a) && right.can_resolve(b) {
        Some((*a.clone(), *b.clone()))
    } else if left.can_resolve(b) && right.can_resolve(a) {
        Some((*b.clone(), *a.clone()))
    } else {
        None
    }
}

fn bind_value(v: &ast::Value) -> Result<ScalarValue> {
    Ok(match v {
        ast::Value::Number(s, _) => parse_number(s)?,
        ast::Value::SingleQuotedString(s) => ScalarValue::Utf8(s.clone()),
        ast::Value::Boolean(b) => ScalarValue::Boolean(*b),
        ast::Value::Null => ScalarValue::Null,
        other => return unsupported(format!("literal {other}")),
    })
}

/// Integers become Int64; `1.25` becomes an exact Decimal(125, 2); exponent
/// notation becomes Float64.
fn parse_number(s: &str) -> Result<ScalarValue> {
    let bad = || MiniLakeError::Plan(format!("invalid number '{s}'"));
    if s.contains(['e', 'E']) {
        return s.parse::<f64>().map(ScalarValue::Float64).map_err(|_| bad());
    }
    if let Some((int, frac)) = s.split_once('.') {
        let digits = format!("{int}{frac}");
        let v: i128 = digits.parse().map_err(|_| bad())?;
        let scale = u8::try_from(frac.len()).map_err(|_| bad())?;
        return Ok(ScalarValue::Decimal(v, scale));
    }
    s.parse::<i64>().map(ScalarValue::Int64).map_err(|_| bad())
}

fn const_usize(e: &ast::Expr) -> Result<usize> {
    match e {
        ast::Expr::Value(v) => match &v.value {
            ast::Value::Number(s, _) => s
                .parse()
                .map_err(|_| MiniLakeError::Plan(format!("invalid LIMIT/OFFSET '{s}'"))),
            other => unsupported(format!("LIMIT {other}")),
        },
        other => unsupported(format!("LIMIT {other}")),
    }
}

/// `ORDER BY 2` -> Some(1)
fn positional(e: &ast::Expr) -> Option<usize> {
    match e {
        ast::Expr::Value(v) => match &v.value {
            ast::Value::Number(s, _) => s.parse::<usize>().ok().and_then(|n| n.checked_sub(1)),
            _ => None,
        },
        _ => None,
    }
}

/// Unqualified identifier name, lower-cased.
fn alias_ref(e: &ast::Expr) -> Option<String> {
    match e {
        ast::Expr::Identifier(id) => Some(id.value.to_ascii_lowercase()),
        _ => None,
    }
}

fn map_type(dt: &ast::DataType) -> Result<DataType> {
    let s = dt.to_string().to_ascii_uppercase();
    let base = s.split('(').next().unwrap_or("").trim();
    Ok(match base {
        "INT" | "INTEGER" | "INT4" | "SMALLINT" | "TINYINT" => DataType::Int32,
        "BIGINT" | "INT8" | "LONG" => DataType::Int64,
        "DOUBLE" | "DOUBLE PRECISION" | "FLOAT" | "FLOAT8" | "REAL" | "DECIMAL" | "NUMERIC" => {
            DataType::Float64
        }
        "DATE" => DataType::Date,
        "VARCHAR" | "TEXT" | "STRING" | "CHAR" => DataType::Utf8,
        "BOOLEAN" | "BOOL" => DataType::Boolean,
        _ => return unsupported(format!("type {s}")),
    })
}

/// `base +/- INTERVAL 'n' unit`.
fn apply_interval(base: Expr, iv: &ast::Interval, sign: i64) -> Result<Expr> {
    let text = match iv.value.as_ref() {
        ast::Expr::Value(v) => match &v.value {
            ast::Value::SingleQuotedString(s) => s.clone(),
            ast::Value::Number(s, _) => s.clone(),
            other => return unsupported(format!("interval value {other}")),
        },
        other => return unsupported(format!("interval value {other}")),
    };
    let mut parts = text.split_whitespace();
    let n: i64 = parts
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| MiniLakeError::Plan(format!("invalid interval '{text}'")))?;
    let unit = match (&iv.leading_field, parts.next()) {
        (Some(f), _) => f.to_string().to_ascii_uppercase(),
        (None, Some(u)) => u.to_ascii_uppercase(),
        (None, None) => return unsupported("interval without unit"),
    };
    let unit = unit.trim_end_matches('S');
    let n = n * sign;
    match unit {
        "DAY" => Ok(Expr::binary(
            base,
            BinaryOp::Add,
            Expr::Literal(ScalarValue::Int32(n as i32)),
        )),
        "WEEK" => Ok(Expr::binary(
            base,
            BinaryOp::Add,
            Expr::Literal(ScalarValue::Int32(n as i32 * 7)),
        )),
        "MONTH" | "YEAR" => {
            let months = if unit == "YEAR" { n * 12 } else { n };
            // Month arithmetic is only supported on constant dates, which is
            // what TPC-H uses; fold the base first.
            match fold_expr(base)? {
                Expr::Literal(ScalarValue::Date(d)) => {
                    Ok(Expr::Literal(ScalarValue::Date(add_months(d, months as i32))))
                }
                _ => unsupported("month/year intervals on non-constant dates"),
            }
        }
        other => unsupported(format!("interval unit {other}")),
    }
}

/// Literal coercion so kernels see matching types without runtime casts of
/// whole columns:
/// * `date_col < '1995-03-15'`   -> string literal becomes a DATE
/// * `int32_col = 5`             -> Int64 literal becomes Int32
/// * `date_col + 90`             -> Int64 literal becomes Int32
pub fn coerce(e: Expr, schema: &LogicalSchema) -> Result<Expr> {
    e.transform(&mut |node| {
        Ok(match node {
            Expr::Binary { op, left, right } if op.is_comparison() || op.is_arithmetic() => {
                let lt = left.data_type(schema).ok();
                let rt = right.data_type(schema).ok();
                let l = coerce_literal(*left, rt)?;
                let r = coerce_literal(*right, lt)?;
                Expr::binary(l, op, r)
            }
            Expr::InList {
                expr,
                list,
                negated,
            } => {
                let t = expr.data_type(schema).ok();
                Expr::InList {
                    list: list
                        .into_iter()
                        .map(|x| coerce_literal(x, t))
                        .collect::<Result<_>>()?,
                    expr,
                    negated,
                }
            }
            other => other,
        })
    })
}

fn coerce_literal(e: Expr, other: Option<DataType>) -> Result<Expr> {
    let Expr::Literal(v) = &e else {
        return Ok(e);
    };
    Ok(match (v, other) {
        (ScalarValue::Utf8(s), Some(DataType::Date)) => {
            Expr::Literal(ScalarValue::Date(parse_date(s).ok_or_else(|| {
                MiniLakeError::Plan(format!("invalid date '{s}'"))
            })?))
        }
        (ScalarValue::Int64(x), Some(DataType::Int32 | DataType::Date)) => {
            match i32::try_from(*x) {
                Ok(v) => Expr::Literal(ScalarValue::Int32(v)),
                Err(_) => e,
            }
        }
        (ScalarValue::Int64(x), Some(DataType::Float64)) => {
            Expr::Literal(ScalarValue::Float64(*x as f64))
        }
        _ => e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(parse_number("24").unwrap(), ScalarValue::Int64(24));
        assert_eq!(parse_number("0.06").unwrap(), ScalarValue::Decimal(6, 2));
        assert_eq!(parse_number("1e3").unwrap(), ScalarValue::Float64(1000.0));
    }
}
