use std::cmp::Ordering;
use std::hash::Hash;

/// A trait for conflict-free replicated data types.
/// A `Crdt` satisfies the following properties:
/// * Multiple parties, each with a copy of the data (replica), can freely update their copies,
///   without needing to coordinate with each other (e.g. lock)
/// * Through some algorithm, parties transmit "reports" to each other.
///   Received reports get merged in an error-free way.
/// * All parties will eventually converge on the same data.
///
/// Currently, this library only supports state-based CRDTs
/// (a.k.a. CvRDTs, convergent replicated data types.)
/// where "reports" contain all the data.
/// In other theoretical CRDT variants, replicas share portions of the data (delta-CRDTs)
/// or commutative operations (CmRDTs.)
pub trait Crdt {
    /// A "report" contains the relevant data for transmission to another replica.
    ///
    /// Generally, the lifecycle of a report is:
    /// * Creation (e.g. through `crdt.report()`)
    /// * Serialized (e.g. through `serde`)
    /// * Transmitted (e.g. over a network)
    /// * Received on a separate replica
    /// * Deserialized
    /// * Merged (e.g. through `crdt.merge(&report)`)
    ///
    /// For simple CRDTs, a Report may be `Self`.
    /// But, some CRDTs may need to report additional fields,
    /// for example, a computed value such as a hash.
    /// Other CRDTs may have private fields,
    /// such as a secret key for authentication,
    /// which should not be included in the report.
    /// In yet other cases, the report may be completely different,
    /// such as an encrypted message.
    type Report;

    /// The result of the `merge()` operation.
    /// `merge()` is infallible--so this isn't for error reporting
    /// and generally shouldn't be `Result`.
    /// Rather, this type communicates what happened during the `merge()`
    /// (e.g. "the value was updated", "the report was outdated")
    /// which may help inform the data-sharing algorithm
    /// or downstream application code.
    ///
    /// A good rule of thumb is: it should always be safe to ignore the `MergeResult`.
    type MergeResult;

    /// Generate a report containing all the data,
    /// typically for transmission to another replica.
    fn report(&self) -> Self::Report;

    /// Merge in a report (e.g. one received from another replica.)
    fn merge(&mut self, report: &Self::Report) -> Self::MergeResult;
}

/// Configuration trait for HMAC hash computation.
///
/// This trait encapsulates the hash algorithm, secret type, and output format.
/// Implementors define how to compute a hash from the secret and data.
pub trait HmacConfig: Clone {
    /// The type of the secret/key (must be hashable)
    type Secret: Hash + Clone;

    /// The type of the hash output (must be comparable)
    type Output: Clone + PartialEq;

    /// Get a reference to the secret
    fn secret(&self) -> &Self::Secret;

    /// Compute the hash of the secret combined with the given data
    fn compute_hash<T: Hash>(&self, data: &T) -> Self::Output;
}

/// A wrapper which implements hash-based message authentication.
///
/// Any received reports will be validated against the secret.
/// If the resulting hash does not match, the inner data will not be updated.
///
/// Note that this provides authentication but not encryption--
/// all the data remains in plaintext.
///
/// The hash algorithm, secret type, and hash output type are all configurable
/// via the `HmacConfig` trait.
#[derive(Clone, Debug)]
pub struct Hmac<T, C: HmacConfig> {
    /// The hash configuration (contains the secret and hash algorithm)
    pub config: C,

    /// The wrapped type (a CRDT whose report can be hashed)
    pub inner: T,
}

/// The report generated from an `Hmac`-wrapped CRDT.
/// This report contains the inner report,
/// plus a hash of its shared secret concatenated with its data.
/// The report does not contain the shared secret itself.
#[derive(Clone, Debug, Hash)]
pub struct HmacReport<R, O> {
    pub inner: R,
    pub hash: O,
}

impl<T: Crdt, C: HmacConfig> Crdt for Hmac<T, C>
where
    T::Report: Hash,
{
    type Report = HmacReport<T::Report, C::Output>;
    type MergeResult = Option<T::MergeResult>;

    /// Computes the hash and returns it along with the inner report.
    fn report(&self) -> Self::Report {
        let inner = self.inner.report();
        let hash = self.config.compute_hash(&inner);
        HmacReport { inner, hash }
    }

    /// Checks the hash and calls `.merge()` on the inner type.
    /// Returns None if the hash does not match and no update was made.
    fn merge(&mut self, other: &Self::Report) -> Self::MergeResult {
        let expected_hash = self.config.compute_hash(&other.inner);
        if other.hash != expected_hash {
            return None;
        }

        Some(self.inner.merge(&other.inner))
    }
}

impl<T, C: HmacConfig> AsRef<T> for Hmac<T, C> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T, C: HmacConfig> AsMut<T> for Hmac<T, C> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T0: Crdt, T1: Crdt> Crdt for (T0, T1) {
    type Report = (T0::Report, T1::Report);
    type MergeResult = (T0::MergeResult, T1::MergeResult);

    fn report(&self) -> Self::Report {
        let (s0, s1) = self;
        (s0.report(), s1.report())
    }

    fn merge(&mut self, other: &Self::Report) -> Self::MergeResult {
        let (s0, s1) = self;
        let (o0, o1) = other;

        (s0.merge(o0), s1.merge(o1))
    }
}

/// A wrapper for restricting what incoming data is acted upon.
///
/// Any received reports must have a `key` that matches this one.
/// If the key does not match, the inner data will not be updated.
///
/// Note that this provides no security whatsoever
/// since the key appears in the report in plaintext.
/// If you want authentication, consider `Hmac`.
#[derive(Clone, Debug, Hash)]
pub struct Filter<K, T> {
    pub key: K,
    pub inner: T,
}

impl<K: Clone + PartialEq, T: Crdt> Crdt for Filter<K, T> {
    type Report = Filter<K, T::Report>;
    type MergeResult = Option<T::MergeResult>;

    fn report(&self) -> Self::Report {
        Filter {
            key: self.key.clone(),
            inner: self.inner.report(),
        }
    }

    /// Checks the key and calls `.merge()` on the inner type if it matches.
    /// Returns None if the key does not match and no update was made.
    fn merge(&mut self, other: &Self::Report) -> Self::MergeResult {
        if self.key != other.key {
            return None;
        }
        Some(self.inner.merge(&other.inner))
    }
}

impl<K, T> AsRef<T> for Filter<K, T> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<K, T> AsMut<T> for Filter<K, T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

/// A CRDT which can only increase.
/// Merge is implementing by keeping whichever data compares greater.
///
/// `T` does not need to be a CRDT, but it needs to implement `Ord`.
///
/// `Max` takes ownership of the inner type,
/// in order to ensure that it cannot decrease.
/// Updates to the inner value should be done using `merge()`.
#[derive(Clone, Debug, Hash)]
pub struct Max<T>(T);

impl<T: Clone + Ord> Crdt for Max<T> {
    type Report = T;
    type MergeResult = Ordering;

    fn report(&self) -> Self::Report {
        self.0.clone()
    }

    /// Compares `other` with `self`. If `other` is greater, its value is assumed.
    /// Otherwise, no update is made.
    /// Returns the result of the comparison:
    ///  * `Ordering::Less`: The report was outdated.
    ///  * `Ordering::Equal`: The report was redundant.
    ///  * `Ordering::Greater`: The report was newer and was assumed.
    fn merge(&mut self, other: &Self::Report) -> Self::MergeResult {
        let result = other.cmp(&self.0);
        if matches!(result, Ordering::Greater) {
            self.0 = other.clone(); // Replace with write method somehow
        }
        result
    }
}

impl<T> AsRef<T> for Max<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use std::hash::{BuildHasher, Hasher};

    /// HMAC configuration using Shake128 with 16-byte secret and 16-byte output
    #[derive(Clone, Debug)]
    struct Shake128Config {
        secret: [u8; 16],
    }

    impl HmacConfig for Shake128Config {
        type Secret = [u8; 16];
        type Output = [u8; 16];

        fn secret(&self) -> &Self::Secret {
            &self.secret
        }

        fn compute_hash<T: Hash>(&self, data: &T) -> Self::Output {
            use rs_shake128::Shake128State;

            // Create a new hasher
            let hasher_builder = Shake128State::<16>::default();
            let mut hasher = hasher_builder.build_hasher();

            // Hash the secret
            self.secret.hash(&mut hasher);

            // Hash the data
            data.hash(&mut hasher);

            // Extract 16 bytes by calling finish() twice
            // (Shake128 allows multiple finish() calls for XOF)
            let mut result = [0_u8; 16];
            result[0..8].copy_from_slice(&hasher.finish().to_le_bytes());
            result[8..16].copy_from_slice(&hasher.finish().to_le_bytes());
            result
        }
    }

    #[test]
    fn test_max_keeps_maximum() {
        let mut max_crdt = Max(5u32);

        // Merging with a smaller value should not update
        assert_eq!(max_crdt.merge(&3), Ordering::Less);
        assert_eq!(max_crdt.as_ref(), &5);

        // Merging with an equal value should not update
        assert_eq!(max_crdt.merge(&5), Ordering::Equal);
        assert_eq!(max_crdt.as_ref(), &5);

        // Merging with a larger value should update
        assert_eq!(max_crdt.merge(&10), Ordering::Greater);
        assert_eq!(max_crdt.as_ref(), &10);
    }

    #[test]
    fn test_filter_requires_matching_keys() {
        let mut filter_crdt = Filter {
            key: "alice",
            inner: Max(100u32),
        };

        // Report with matching key should merge successfully
        let matching_report = Filter {
            key: "alice",
            inner: 150u32,
        };
        assert_eq!(filter_crdt.merge(&matching_report), Some(Ordering::Greater));
        assert_eq!(filter_crdt.as_ref().as_ref(), &150);

        // Report with non-matching key should be rejected
        let mismatched_report = Filter {
            key: "bob",
            inner: 200u32,
        };
        assert_eq!(filter_crdt.merge(&mismatched_report), None);
        assert_eq!(filter_crdt.as_ref().as_ref(), &150);
    }

    #[test]
    fn test_hmac_validates_hash() {
        let config = Shake128Config {
            secret: [0xBB; 16],
        };

        let mut hmac_crdt = Hmac {
            config: config.clone(),
            inner: Max(42u32),
        };

        // Create a valid report from a replica
        let replica = Hmac {
            config: config.clone(),
            inner: Max(50u32),
        };
        let valid_report = replica.report();

        // Valid report should merge successfully
        assert_eq!(hmac_crdt.merge(&valid_report), Some(Ordering::Greater));
        assert_eq!(hmac_crdt.as_ref().as_ref(), &50);

        // Invalid report (manually crafted with wrong hash)
        let invalid_report = HmacReport {
            inner: 100u32,
            hash: [0xDE; 16],
        };
        assert_eq!(hmac_crdt.merge(&invalid_report), None);
        assert_eq!(hmac_crdt.as_ref().as_ref(), &50);
    }

    #[test]
    fn test_tuple_crdt() {
        let mut tuple_crdt = (Max(10u32), Max("alice"));

        let report = (15u32, "bob");
        let result = tuple_crdt.merge(&report);

        assert_eq!(result, (Ordering::Greater, Ordering::Greater));
        assert_eq!(tuple_crdt.0.as_ref(), &15);
        assert_eq!(tuple_crdt.1.as_ref(), &"bob");
    }

    #[test]
    fn test_as_ref_as_mut() {
        let config = Shake128Config {
            secret: [0x12; 16],
        };

        let mut hmac_crdt = Hmac {
            config,
            inner: Max(100u32),
        };

        // Test AsRef
        let inner_ref: &Max<u32> = hmac_crdt.as_ref();
        assert_eq!(inner_ref.as_ref(), &100);

        // Test AsMut
        let inner_mut: &mut Max<u32> = hmac_crdt.as_mut();
        inner_mut.merge(&200);
        assert_eq!(hmac_crdt.inner.as_ref(), &200);
    }

    #[test]
    fn test_nested_crdts() {
        let config = Shake128Config {
            secret: [0xAB; 16],
        };

        let mut nested = Hmac {
            config: config.clone(),
            inner: Filter {
                key: 123,
                inner: Max(0u32),
            },
        };

        // Create a report from a replica with updated value
        let mut replica = nested.clone();
        replica.as_mut().as_mut().merge(&50);

        let report = replica.report();
        let result = nested.merge(&report);

        assert!(result.is_some());
        assert_eq!(nested.as_ref().as_ref().as_ref(), &50);
    }

    #[test]
    fn test_shake128_different_secrets() {
        // Create two CRDTs with different secrets
        let config_alice = Shake128Config {
            secret: *b"alice_secret_key",
        };
        let config_bob = Shake128Config {
            secret: *b"bob_s_secret_key",
        };

        let mut alice_crdt = Hmac {
            config: config_alice.clone(),
            inner: Max(100u32),
        };

        let mut bob_crdt = Hmac {
            config: config_bob.clone(),
            inner: Max(200u32),
        };

        // Bob's report should be rejected by Alice (different secret)
        let bob_report = bob_crdt.report();
        assert_eq!(alice_crdt.merge(&bob_report), None);
        assert_eq!(alice_crdt.as_ref().as_ref(), &100);

        // Alice's report should be rejected by Bob (different secret)
        let alice_report = alice_crdt.report();
        assert_eq!(bob_crdt.merge(&alice_report), None);
        assert_eq!(bob_crdt.as_ref().as_ref(), &200);

        // Create a replica of Alice with the same secret
        let alice_replica = Hmac {
            config: config_alice.clone(),
            inner: Max(150u32),
        };

        // Alice's replica report should be accepted by Alice
        let replica_report = alice_replica.report();
        assert_eq!(alice_crdt.merge(&replica_report), Some(Ordering::Greater));
        assert_eq!(alice_crdt.as_ref().as_ref(), &150);
    }
}
