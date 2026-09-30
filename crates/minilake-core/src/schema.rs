//! Column names and types.

use std::fmt;
use std::sync::Arc;

use crate::DataType;

/// One column of a schema.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    /// Column name.
    pub name: String,
    /// Column type.
    pub data_type: DataType,
    /// May contain NULLs.
    pub nullable: bool,
}

impl Field {
    /// New field.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Field {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

/// Ordered list of fields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Schema {
    /// Fields in column order.
    pub fields: Vec<Field>,
}

/// Shared schema.
pub type SchemaRef = Arc<Schema>;

impl Schema {
    /// New schema.
    pub fn new(fields: Vec<Field>) -> Self {
        Schema { fields }
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// True when there are no columns.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Index of the column named `name`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    /// Schema with only the columns at `indices`.
    pub fn project(&self, indices: &[usize]) -> Schema {
        Schema::new(indices.iter().map(|&i| self.fields[i].clone()).collect())
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .fields
            .iter()
            .map(|x| format!("{}:{}", x.name, x.data_type))
            .collect();
        write!(f, "[{}]", parts.join(", "))
    }
}
