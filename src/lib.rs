use std::cmp::Ordering;

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

/// A wrapper which implements hash-based message authentication.
///
/// Any received reports will be validated against the secret.
/// If the resulting hash does not match, the inner data will not be updated.
///
/// Note that this provides authentication but not encryption--
/// all the data remains in plaintext.
#[derive(Clone, Debug)]
pub struct Hmac<T> {
    /// The shared secret (this will NOT be included in the report)
    secret: u64,

    /// The wrapped type (a CRDT whose report can be hashed)
    inner: T,
}

/// The report generated from an `Hmac`-wrapped CRDT.
/// This report contains the inner report,
/// plus a hash of its shared secret concatenated with its data.
/// The report does not contain the shared secret itself.
#[derive(Clone, Debug)]
pub struct HmacReport<T> {
    inner: T,
    hash: u64,
}

impl<T: Crdt> Crdt for Hmac<T> {
    type Report = HmacReport<T::Report>;
    type MergeResult = Option<T::MergeResult>;

    /// Computes the hash and returns it along with the inner report.
    fn report(&self) -> Self::Report {
        let inner = self.inner.report();
        let hash = 0xAAAA; // TODO: hash inner with self.private
        HmacReport { inner, hash }
    }

    /// Checks the hash and calls `.merge()` on the inner type.
    /// Returns None if the hash does not match and no update was made.
    fn merge(&mut self, other: &Self::Report) -> Self::MergeResult {
        if other.hash != 0xAAAA {
            return None;
        }

        Some(self.inner.merge(&other.inner))
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
#[derive(Clone, Debug)]
pub struct Filter<K, T> {
    key: K,
    inner: T,
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

/// A CRDT which can only increase.
/// Merge is implementing by keeping whichever data compares greater.
///
/// `T` does not need to be a CRDT, but it needs to implement `Ord`.
///
/// `Max` takes ownership of the inner type,
/// in order to ensure that it cannot decrease.
/// Updates to the inner value should be done using `merge()`.
#[derive(Clone, Debug)]
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

#[cfg(test)]
mod test {
    use crate::*;

    #[derive(Debug, Default, PartialOrd, PartialEq, Ord, Eq, Clone)]
    struct MyVal {
        a: u32,
        b: u32,
    }

    #[test]
    fn test() {
        let mut my_crdt = Hmac {
            secret: 0xBBBB,
            inner: Filter {
                key: 0x1234,
                inner: Max(MyVal { a: 0, b: 0 }),
            },
        };

        let mut my_crdt_2 = my_crdt.clone();

        my_crdt_2.inner.inner.merge(&MyVal { a: 2, b: 0 });
        my_crdt_2.inner.inner.merge(&MyVal { a: 5, b: 0 });

        dbg!(&my_crdt_2);

        let updated_report = my_crdt_2.report();
        dbg!(&updated_report);

        let merge_result = my_crdt.merge(&updated_report);
        let consistent = merge_result
            .flatten()
            .map(|x| x == Ordering::Equal)
            .unwrap_or(true);
        dbg!(consistent);
    }
}
