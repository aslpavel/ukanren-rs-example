/// Very simple uKanren implementation, with einstien problem as unittest
use std::fmt;
use std::iter;
use std::rc::Rc;

// -----------------------------------------------------------------------------
// Val and Var
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var(u64);

#[derive(Debug, PartialEq, Eq)]
enum ValInner {
    Int(i64),
    Str(String),
    Var(Var),
    Pair(Val, Val),
    Nil,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Val(Rc<ValInner>);

impl Val {
    pub fn nil() -> Self {
        Val(Rc::new(ValInner::Nil))
    }
}

impl fmt::Debug for Val {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.as_ref() {
            &ValInner::Int(int) => write!(f, "{}", int),
            &ValInner::Str(ref string) => write!(f, "\"{}\"", string),
            &ValInner::Var(index) => write!(f, "_{}", index.0),
            &ValInner::Nil => write!(f, "Nil"),
            &ValInner::Pair(..) => {
                let mut rest = self.clone();
                let mut first = true;
                write!(f, "(")?;
                while let &ValInner::Pair(ref head, ref tail) = rest.clone().as_ref() {
                    if first {
                        first = false;
                    } else {
                        write!(f, " ")?;
                    }
                    rest = tail.clone();
                    head.fmt(f)?;
                }
                match rest.as_ref() {
                    &ValInner::Nil => (),
                    _ => {
                        write!(f, " . ")?;
                        rest.fmt(f)?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}

impl AsRef<ValInner> for Val {
    fn as_ref(&self) -> &ValInner {
        self.0.as_ref()
    }
}

impl<'a> From<&'a Val> for Val {
    fn from(value: &Val) -> Val {
        value.clone()
    }
}

impl From<i64> for Val {
    fn from(value: i64) -> Val {
        Val(Rc::new(ValInner::Int(value)))
    }
}

impl From<String> for Val {
    fn from(value: String) -> Val {
        Val(Rc::new(ValInner::Str(value)))
    }
}

impl<'a> From<&'a str> for Val {
    fn from(value: &'a str) -> Val {
        Val(Rc::new(ValInner::Str(value.into())))
    }
}

impl From<Var> for Val {
    fn from(value: Var) -> Val {
        Val(Rc::new(ValInner::Var(value)))
    }
}

impl<A, B> From<(A, B)> for Val
where
    A: Into<Val>,
    B: Into<Val>,
{
    fn from((a, b): (A, B)) -> Val {
        Val(Rc::new(ValInner::Pair(a.into(), b.into())))
    }
}

fn unify(smap: SMap, left: &Val, right: &Val) -> Option<SMap> {
    let left = smap.walk(left);
    let right = smap.walk(right);

    match (left.as_ref(), right.as_ref()) {
        (&ValInner::Var(ref left_var), &ValInner::Var(ref right_var)) if left_var == right_var => {
            Some(smap)
        }
        (&ValInner::Var(ref left_var), _) => Some(smap.assoc(*left_var, right.clone())),
        (_, &ValInner::Var(ref right_var)) => Some(smap.assoc(*right_var, left.clone())),
        (&ValInner::Pair(ref p00, ref p01), &ValInner::Pair(ref p10, ref p11)) => {
            unify(smap, &p00, &p10).and_then(|smap| unify(smap, &p01, &p11))
        }
        _ if left == right => Some(smap),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// Substitution Map
// -----------------------------------------------------------------------------
enum SMapInner {
    Empty,
    KeyValue { key: Var, value: Val, rest: SMap },
}

#[derive(Clone)]
pub struct SMap(Rc<SMapInner>);
struct SMapIter(SMap);

impl SMap {
    fn new() -> Self {
        SMap(Rc::new(SMapInner::Empty))
    }

    fn assoc(&self, key: Var, value: Val) -> Self {
        SMap(Rc::new(SMapInner::KeyValue {
            key,
            value,
            rest: self.clone(),
        }))
    }

    fn walk(&self, key: &Val) -> Val {
        match *key.0 {
            ValInner::Var(var) => {
                if let Some((_, value)) = self.iter().find(|(ikey, _)| *ikey == var) {
                    self.walk(&value)
                } else {
                    key.clone()
                }
            }
            _ => key.clone(),
        }
    }

    fn deep_walk(&self, val: &Val) -> Val {
        let val = self.walk(val);
        match *val.0 {
            ValInner::Var(_) => val.clone(),
            ValInner::Pair(ref val0, ref val1) => {
                (self.deep_walk(&val0), self.deep_walk(&val1)).into()
            }
            _ => val.clone(),
        }
    }

    fn iter(&self) -> SMapIter {
        SMapIter(self.clone())
    }
}

impl fmt::Debug for SMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:#?}",
            self.iter()
                .map(|(Var(var), val)| (format!("_{}", var), val))
                .collect::<std::collections::HashMap<_, _>>()
        )
    }
}

impl Iterator for SMapIter {
    type Item = (Var, Val);

    fn next(&mut self) -> Option<Self::Item> {
        let (item, rest) = match *(self.0).0 {
            SMapInner::Empty => return None,
            SMapInner::KeyValue {
                key,
                ref value,
                ref rest,
            } => (Some((key, value.clone())), rest.clone()),
        };
        self.0 = rest;
        item
    }
}

// -----------------------------------------------------------------------------
// Stream
// -----------------------------------------------------------------------------
pub struct Stream<T> {
    iter: Box<Iterator<Item = T>>,
}

impl<T> Stream<T>
where
    T: 'static,
{
    fn zero() -> Self {
        Stream {
            iter: Box::new(iter::empty()),
        }
    }

    fn unit(value: T) -> Self {
        Stream {
            iter: Box::new(iter::once(value)),
        }
    }

    fn plus(self, other: Self) -> Self {
        // TODO: interleave streams?
        Stream {
            iter: Box::new(self.iter.chain(other.iter)),
        }
    }

    fn bind<F>(self, mut f: F) -> Self
    where
        F: FnMut(T) -> Self + 'static,
    {
        Stream {
            iter: Box::new(self.iter.flat_map(move |item| f(item).iter)),
        }
    }
}

impl<T> IntoIterator for Stream<T> {
    type Item = T;
    type IntoIter = Box<Iterator<Item = T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }
}

// -----------------------------------------------------------------------------
// Goals
// -----------------------------------------------------------------------------
pub type State = (SMap, u64);
pub type Goal = Rc<Fn(State) -> Stream<State>>;

pub fn eq<L: Into<Val>, R: Into<Val>>(left: L, right: R) -> Goal {
    let left = left.into();
    let right = right.into();
    Rc::new(move |(smap, count)| match unify(smap, &left, &right) {
        None => Stream::zero(),
        Some(smap) => Stream::unit((smap, count)),
    })
}

pub fn succ() -> Goal {
    Rc::new(|state| Stream::unit(state))
}

pub fn fail() -> Goal {
    Rc::new(|_| Stream::zero())
}

pub fn conj(first: Goal, second: Goal) -> Goal {
    Rc::new(move |state| {
        first(state).bind({
            let second = second.clone();
            move |state| second(state)
        })
    })
}

pub fn disj(first: Goal, second: Goal) -> Goal {
    Rc::new(move |state| first(state.clone()).plus(second(state)))
}

pub fn fresh<Body>(body: Body) -> Goal
where
    Body: Fn(Var) -> Goal + 'static,
{
    Rc::new(move |(smap, count)| body(Var(count))((smap, count + 1)))
}

pub fn run<Body>(body: Body) -> impl Iterator<Item = Val>
where
    Body: Fn(Var) -> Goal,
{
    (body)(Var(0))((SMap::new(), 1))
        .into_iter()
        .map(|(smap, _)| smap.deep_walk(&Var(0).into()))
}

// -----------------------------------------------------------------------------
// Macros
// -----------------------------------------------------------------------------
#[macro_export]
macro_rules! fresh {
    (| $($var:ident),* | $body:expr) => {{
        let goal: $crate::Goal = Rc::new(move |(smap, count)| {
            let mut count = count;
            $(
                let $var = Var(count);
                count += 1;
            )*
            ($body)((smap, count))
        });
        goal
    }};
}

#[macro_export]
macro_rules! conj {
    ($goal:expr, $($goals:expr),+ $(,)*) => {
        conj($goal, conj!($($goals,)*))
    };
    ($goal:expr $(,)*) => { $goal };
}

#[macro_export]
macro_rules! disj {
    ($goal:expr, $($goals:expr),+ $(,)*) => {
        disj($goal, disj!($($goals,)*))
    };
    ($goal:expr $(,)*) => { $goal };
}

#[macro_export]
macro_rules! list {
    ($($val:expr),* $(,)*) => {{
        let slice: Box<[Val]> = Box::new([$($val.into(),)*]);
        slice
            .into_vec()
            .into_iter()
            .rev()
            .fold(Val::nil(), |acc, val| Val::from((val, acc)))
    }};
}

#[cfg(test)]
mod test {
    use super::*;

    fn left_of(i0: &Val, i1: &Val) -> Goal {
        (0..4)
            .map(|index| conj(eq(i0.clone(), index), eq(i1.clone(), index + 1)))
            .fold(fail(), disj)
    }

    fn next_to(i0: &Val, i1: &Val) -> Goal {
        disj(left_of(i0, i1), left_of(i1, i0))
    }

    fn member(x: &Val, xs: &Val) -> Goal {
        let x = x.clone();
        let xs = xs.clone();
        fresh! {
            |x0, x1, x2, x3, x4| conj! {
                eq(&xs, list!(x0, x1, x2, x3, x4)),
                disj! {
                    eq(&x, x0),
                    eq(&x, x1),
                    eq(&x, x2),
                    eq(&x, x3),
                    eq(&x, x4),
                }
            }
        }
    }

    #[derive(Clone)]
    struct House {
        index: Option<Val>,
        nation: Option<Val>,
        color: Option<Val>,
        pet: Option<Val>,
        drink: Option<Val>,
        smoke: Option<Val>,
    }

    impl House {
        fn new() -> Self {
            House {
                index: None,
                nation: None,
                color: None,
                pet: None,
                drink: None,
                smoke: None,
            }
        }

        fn index<V: Into<Val>>(&self, index: V) -> Self {
            let mut house = self.clone();
            house.index = Some(index.into());
            house
        }

        fn nation<V: Into<Val>>(&self, nation: V) -> Self {
            let mut house = self.clone();
            house.nation = Some(nation.into());
            house
        }

        fn color<V: Into<Val>>(&self, color: V) -> Self {
            let mut house = self.clone();
            house.color = Some(color.into());
            house
        }

        fn pet<V: Into<Val>>(&self, pet: V) -> Self {
            let mut house = self.clone();
            house.pet = Some(pet.into());
            house
        }

        fn drink<V: Into<Val>>(&self, drink: V) -> Self {
            let mut house = self.clone();
            house.drink = Some(drink.into());
            house
        }

        fn smoke<V: Into<Val>>(&self, smoke: V) -> Self {
            let mut house = self.clone();
            house.smoke = Some(smoke.into());
            house
        }

        fn eq<V: Into<Val>>(&self, house: V) -> Goal {
            let this = self.clone();
            let house = house.into().clone();
            fresh! {
                |findex, fnation, fcolor, fpet, fdrink, fsmoke| {
                    let House { index, nation, color, pet, drink, smoke } = this.clone();
                    let mut goals = Vec::new();
                    if let Some(index) = index {
                        goals.push(eq(index, findex));
                    }
                    if let Some(nation) = nation {
                        goals.push(eq(nation, fnation));
                    }
                    if let Some(color) = color {
                        goals.push(eq(color, fcolor));
                    }
                    if let Some(pet) = pet {
                        goals.push(eq(pet, fpet));
                    }
                    if let Some(drink) = drink {
                        goals.push(eq(drink, fdrink));
                    }
                    if let Some(smoke) = smoke {
                        goals.push(eq(smoke, fsmoke));
                    }
                    conj(
                        eq(house.clone(), list!(findex, fnation, fcolor, fpet, fdrink, fsmoke)),
                        goals.into_iter().fold(succ(), conj),
                    )
                }
            }
        }
    }

    fn house(street: &Val, house: &Val, attrs: &House) -> Goal {
        conj(attrs.eq(house), member(house, street))
    }

    fn house_exists(street: &Val, attrs: &House) -> Goal {
        let street = street.clone();
        let attrs = attrs.clone();
        fresh! {|h| house(&street, &h.into(), &attrs)}
    }

    fn house_next_to(street: &Val, attrs_0: &House, attrs_1: &House) -> Goal {
        let street = street.clone();
        let attrs_0 = attrs_0.clone();
        let attrs_1 = attrs_1.clone();
        fresh! {
            |h0, i0, h1, i1|
            conj! {
                next_to(&i0.into(), &i1.into()),
                house(&street, &h0.into(), &attrs_0.index(i0)),
                house(&street, &h1.into(), &attrs_1.index(i1)),
            }
        }
    }

    fn house_left_of(street: &Val, attrs_0: &House, attrs_1: &House) -> Goal {
        let street = street.clone();
        let attrs_0 = attrs_0.clone();
        let attrs_1 = attrs_1.clone();
        fresh! {
            |h0, i0, h1, i1|
            conj! {
                left_of(&i0.into(), &i1.into()),
                house(&street, &h0.into(), &attrs_0.index(i0)),
                house(&street, &h1.into(), &attrs_1.index(i1)),
            }
        }
    }

    fn street_indices(street: &Val) -> Goal {
        let street = street.clone();
        fresh! {
            |h0, h1, h2, h3, h4|
            conj! {
                eq(&street, list!(h0, h1, h2, h3, h4)),
                House::new().index(0).eq(h0),
                House::new().index(1).eq(h1),
                House::new().index(2).eq(h2),
                House::new().index(3).eq(h3),
                House::new().index(4).eq(h4),
            }
        }
    }

    fn einstein_problem(result: Var) -> Goal {
        fresh! {
            |fish_owner, street| {
                let street = Val::from(street);
                conj! {
                    eq(result, (fish_owner, &street)),
                    street_indices(&street),
                    house_exists(&street, &House::new().nation("brit").color("red")),
                    house_exists(&street, &House::new().nation("swede").pet("dog")),
                    house_exists(&street, &House::new().nation("dane").drink("tea")),
                    house_left_of(&street, &House::new().color("green"), &House::new().color("white")),
                    house_exists(&street, &House::new().color("green").drink("coffee")),
                    house_exists(&street, &House::new().smoke("pall mall").pet("bird")),
                    house_exists(&street, &House::new().smoke("dunhill").color("yellow")),
                    house_exists(&street, &House::new().index(2).drink("milk")),
                    house_exists(&street, &House::new().index(0).nation("norweigan")),
                    house_next_to(&street, &House::new().smoke("blend"), &House::new().pet("cat")),
                    house_next_to(&street, &House::new().pet("horse"), &House::new().smoke("dunhill")),
                    house_exists(&street, &House::new().drink("beer").smoke("bluemaster")),
                    house_exists(&street, &House::new().nation("german").smoke("prince")),
                    house_next_to(&street, &House::new().nation("norweigan"), &House::new().color("blue")),
                    house_next_to(&street, &House::new().smoke("blend"), &House::new().drink("water")),
                    house_exists(&street, &House::new().nation(fish_owner).pet("fish")),
                }
            }
        }
    }

    #[test]
    fn einstein_problem_test() {
        let mut iter = run(einstein_problem).into_iter();
        let expected: Val = (
            "german",
            list!(
                list!(0, "norweigan", "yellow", "cat", "water", "dunhill"),
                list!(1, "dane", "blue", "horse", "tea", "blend"),
                list!(2, "brit", "red", "bird", "milk", "pall mall"),
                list!(3, "german", "green", "fish", "coffee", "prince"),
                list!(4, "swede", "white", "dog", "beer", "bluemaster"),
            ),
        ).into();
        if let Some(result) = iter.next() {
            assert_eq!(expected, result);
        }
        assert!(iter.next().is_none())
    }
}
