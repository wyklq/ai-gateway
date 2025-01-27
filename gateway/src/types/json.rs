use std::{fmt, marker::PhantomData};

use serde::{
    de::{Error as DeError, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_with::{de::DeserializeAsWrap, ser::SerializeAsWrap, DeserializeAs, Same, SerializeAs};

pub struct JsonStringCond<T = Same>(PhantomData<T>);

impl<T, TAs> SerializeAs<T> for JsonStringCond<TAs>
where
    TAs: SerializeAs<T>,
{
    fn serialize_as<S>(source: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as an object
        SerializeAsWrap::<T, TAs>::new(source).serialize(serializer)
    }
}

impl<'de, T, TAs> DeserializeAs<'de, T> for JsonStringCond<TAs>
where
    TAs: for<'a> DeserializeAs<'a, T>,
    T: Deserialize<'de>,
{
    fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Helper<S, SAs>(PhantomData<(S, SAs)>);

        impl<'de, S, SAs> Visitor<'de> for Helper<S, SAs>
        where
            SAs: for<'a> DeserializeAs<'a, S>,
            S: Deserialize<'de>,
        {
            type Value = S;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("valid JSON object or string")
            }

            fn visit_map<M>(self, map: M) -> Result<S, M::Error>
            where
                M: MapAccess<'de>,
            {
                DeserializeAsWrap::<S, SAs>::deserialize(
                    serde::de::value::MapAccessDeserializer::new(map),
                )
                .map(DeserializeAsWrap::<S, SAs>::into_inner)
            }

            fn visit_str<E>(self, value: &str) -> Result<S, E>
            where
                E: DeError,
            {
                serde_json::from_str(value)
                    .map(DeserializeAsWrap::<S, SAs>::into_inner)
                    .map_err(DeError::custom)
            }

            fn visit_seq<A>(self, seq: A) -> Result<S, A::Error>
            where
                A: SeqAccess<'de>,
            {
                DeserializeAsWrap::<S, SAs>::deserialize(
                    serde::de::value::SeqAccessDeserializer::new(seq),
                )
                .map(DeserializeAsWrap::<S, SAs>::into_inner)
            }
        }

        deserializer.deserialize_any(Helper::<T, TAs>(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use serde_with::serde_as;

    use super::JsonStringCond;
    #[test]
    fn serialize_object() {
        #[serde_as]
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct Test {
            #[serde_as(as = "JsonStringCond")]
            inner: Inner,
        }

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct Inner {
            i: bool,
        }

        let test = Test {
            inner: Inner { i: true },
        };

        let str = serde_json::to_string(&test).unwrap();
        println!("{str}");

        let json1 = json!({"inner":{"i":true}});
        let obj: Test = serde_json::from_value(json1).unwrap();
        assert_eq!(obj, test);

        let json1 = json!({"inner":"{\"i\":true}"});
        let obj: Test = serde_json::from_value(json1).unwrap();
        assert_eq!(obj, test);
    }
}
