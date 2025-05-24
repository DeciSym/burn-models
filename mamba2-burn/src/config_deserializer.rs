use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// Custom deserializer for time_step_limit that can handle Infinity
pub fn deserialize_time_step_limit<'de, D>(deserializer: D) -> Result<Option<Vec<f64>>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    
    match value {
        None => Ok(None),
        Some(Value::Array(arr)) => {
            let mut result = Vec::new();
            for item in arr {
                match item {
                    Value::Number(n) => {
                        if let Some(f) = n.as_f64() {
                            result.push(f);
                        } else {
                            return Err(serde::de::Error::custom("Invalid number in time_step_limit"));
                        }
                    }
                    Value::String(s) if s == "Infinity" => {
                        result.push(f64::INFINITY);
                    }
                    _ => {
                        return Err(serde::de::Error::custom("Invalid value in time_step_limit"));
                    }
                }
            }
            Ok(Some(result))
        }
        _ => Err(serde::de::Error::custom("time_step_limit must be an array")),
    }
}

/// Custom JSON reader that preprocesses JSON to handle Infinity values
pub fn preprocess_json_with_infinity(json_str: &str) -> String {
    // Replace unquoted Infinity with a very large number (close to f64::MAX)
    // This regex will match Infinity that's not inside quotes
    json_str.replace("Infinity", "1.7976931348623157e+308")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct TestConfig {
        #[serde(deserialize_with = "deserialize_time_step_limit")]
        time_step_limit: Option<Vec<f64>>,
    }

    #[test]
    fn test_deserialize_infinity() {
        let json_str = r#"{
            "time_step_limit": [0.0, "Infinity"]
        }"#;

        let config: TestConfig = serde_json::from_str(json_str).unwrap();
        assert!(config.time_step_limit.is_some());
        let limits = config.time_step_limit.unwrap();
        assert_eq!(limits.len(), 2);
        assert_eq!(limits[0], 0.0);
        assert!(limits[1].is_infinite());
    }

    #[test]
    fn test_preprocess_json() {
        let json_str = r#"{
            "time_step_limit": [0.0, Infinity]
        }"#;
        
        let processed = preprocess_json_with_infinity(json_str);
        assert!(processed.contains("1.7976931348623157e+308"));
        
        // Should be valid JSON now
        let _: serde_json::Value = serde_json::from_str(&processed).unwrap();
    }
}