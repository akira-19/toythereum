use primitive_types::U256;
use std::{cmp::Ordering, collections::HashMap, io::Read, ops::ControlFlow};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1, none_of},
    combinator::{map_res, opt, recognize},
    error::ParseError,
    multi::{fold_many0, many0, separated_list0},
    number::complete::recognize_float,
    sequence::{delimited, pair, preceded, terminated},
    Finish, IResult, Parser,
};

pub fn parse() {
    println!("parse here");
    let mut buf = String::new();
    if !std::io::stdin().read_to_string(&mut buf).is_ok() {
        panic!("Failed to read from stdin");
    }
    println!("{:?}", buf);

    let parsed_statements = match statements_finish(&buf) {
        Ok(parsed_statements) => parsed_statements,
        Err(e) => {
            eprintln!("Parse error: {e:?}");
            return;
        }
    };

    print!("{:?}", parsed_statements);

    let mut frame = StackFrame::new();

    eval_stmts(&parsed_statements, &mut frame);
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    U256(U256),
    Str(String),
    Bool(bool),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U256(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Value::*;
        match (self, other) {
            (U256(lhs), U256(rhs)) => lhs.partial_cmp(rhs),
            _ => None,
        }
    }
}

// impl Value {
//     fn as_i64(&self) -> Option<i64> {
//         match self {
//             Self::U256(val) => Some(*val),
//             Self::Str(val) => val.parse().ok(),
//         }
//     }
// }

pub(crate) fn binary_op_uint(
    lhs: &Value,
    rhs: &Value,
    d: impl Fn(U256, U256) -> U256,
    // i: impl Fn(i64, i64) -> i64,
) -> Value {
    use Value::*;
    match (lhs, rhs) {
        (U256(lhs), U256(rhs)) => U256(d(*lhs, *rhs)),
        _ => {
            panic!("Unsupported operator between {:?} and {:?}", lhs, rhs)
        }
    }
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        binary_op_uint(&self, &rhs, |lhs, rhs| lhs + rhs)
    }
}

impl std::ops::Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_op_uint(&self, &rhs, |lhs, rhs| lhs - rhs)
    }
}

impl std::ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_op_uint(&self, &rhs, |lhs, rhs| lhs * rhs)
    }
}

impl std::ops::Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        binary_op_uint(&self, &rhs, |lhs, rhs| lhs / rhs)
    }
}

enum FnDef<'src> {
    User(UserFn<'src>),
    Native(NativeFn),
}

impl<'src> FnDef<'src> {
    fn call(&self, args: &[Value], frame: &StackFrame) -> Value {
        match self {
            Self::User(code) => {
                let mut new_frame = StackFrame::push_stack(frame);
                new_frame.vars = args
                    .iter()
                    .zip(code.args.iter())
                    .map(|(arg, name)| (name.to_string(), arg.clone()))
                    .collect();
                match eval_stmts(&code.stmts, &mut new_frame) {
                    EvalResult::Continue(val) | EvalResult::Break(BreakResult::Return(val)) => val,
                }
            }
            Self::Native(code) => (code.code)(args),
        }
    }
}

struct UserFn<'src> {
    args: Vec<&'src str>,
    stmts: Statements<'src>,
}

struct NativeFn {
    code: Box<dyn Fn(&[Value]) -> Value>,
}

type Variables = HashMap<String, Value>;
type Functions<'src> = HashMap<String, FnDef<'src>>;

struct StackFrame<'src> {
    vars: Variables,
    funcs: Functions<'src>,
    uplevel: Option<&'src StackFrame<'src>>,
}

impl<'src> StackFrame<'src> {
    fn new() -> Self {
        let mut funcs = Functions::new();
        // funcs.insert("sqrt".to_string(), unary_fn(f64::sqrt));

        Self {
            vars: Variables::new(),
            funcs,
            uplevel: None,
        }
    }

    fn push_stack(uplevel: &'src Self) -> Self {
        Self {
            vars: HashMap::new(),
            funcs: HashMap::new(),
            uplevel: Some(uplevel),
        }
    }

    fn get_fn(&self, name: &str) -> Option<&FnDef<'src>> {
        let mut next_frame = Some(self);
        while let Some(frame) = next_frame {
            if let Some(func) = frame.funcs.get(name) {
                return Some(func);
            }
            next_frame = frame.uplevel;
        }
        None
    }
}

fn eval_stmts<'src>(stmts: &[Statement<'src>], frame: &mut StackFrame<'src>) -> EvalResult {
    let mut last_result = EvalResult::Continue(Value::U256(U256::zero()));
    for statement in stmts {
        match statement {
            Statement::Expression(expr) => {
                last_result = EvalResult::Continue(eval(expr, frame)?);
            }
            Statement::VarDef(name, expr) => {
                let value = eval(expr, frame)?;
                frame.vars.insert(name.to_string(), value);
            }
            Statement::VarAssign(name, expr) => {
                if !frame.vars.contains_key(*name) {
                    panic!("Variable is not defined");
                }
                let value = eval(expr, frame)?;
                frame.vars.insert(name.to_string(), value);
            }

            Statement::FnDef { name, args, stmts } => {
                frame.funcs.insert(
                    name.to_string(),
                    FnDef::User(UserFn {
                        args: args.clone(),
                        stmts: stmts.clone(),
                    }),
                );
            }
            Statement::Return(expr) => {
                return EvalResult::Break(BreakResult::Return(eval(expr, frame)?));
            }
        }
    }
    last_result
}

#[derive(Debug, PartialEq, Clone)]
enum Expression<'src> {
    Ident(&'src str),
    NumLiteral(U256),
    StrLiteral(String),
    BoolLiteral(bool),
    FnInvoke(&'src str, Vec<Expression<'src>>),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
    Sub(Box<Expression<'src>>, Box<Expression<'src>>),
    Mul(Box<Expression<'src>>, Box<Expression<'src>>),
    Div(Box<Expression<'src>>, Box<Expression<'src>>),
    Gt(Box<Expression<'src>>, Box<Expression<'src>>),
    Lt(Box<Expression<'src>>, Box<Expression<'src>>),
    If(
        Box<Expression<'src>>,
        Box<Statements<'src>>,
        Option<Box<Statements<'src>>>,
    ),
}

#[derive(Debug, PartialEq, Clone)]
enum Statement<'src> {
    Expression(Expression<'src>),
    VarDef(&'src str, Expression<'src>),
    VarAssign(&'src str, Expression<'src>),
    FnDef {
        name: &'src str,
        args: Vec<&'src str>,
        stmts: Statements<'src>,
    },
    Return(Expression<'src>),
}

type Statements<'a> = Vec<Statement<'a>>;

// fn unary_fn<'a>(f: fn(f64) -> f64) -> FnDef<'a> {
//     FnDef::Native(NativeFn {
//         code: Box::new(move |args| {
//             Value::F64(f(coerce_f64(
//                 args.into_iter().next().expect("function missing argument"),
//             )))
//         }),
//     })
// }

// fn binary_fn<'a>(f: fn(f64, f64) -> f64) -> FnDef<'a> {
//     FnDef::Native(NativeFn {
//         code: Box::new(move |args| {
//             let mut args = args.into_iter();
//             let lhs = coerce_f64(args.next().expect("function missing the first argument"));
//             let rhs = coerce_f64(args.next().expect("function missing the second argument"));
//             Value::F64(f(lhs, rhs))
//         }),
//     })
// }

#[derive(Debug)]
enum BreakResult {
    Return(Value),
    // Break,
    // Continue,
}

type EvalResult = ControlFlow<BreakResult, Value>;

fn eval<'src>(expr: &Expression<'src>, frame: &mut StackFrame<'src>) -> EvalResult {
    use Expression::*;
    let res = match expr {
        // Ident("pi") => Value::F64(std::f64::consts::PI),
        Ident(id) => frame
            .vars
            .get(*id)
            .cloned()
            .expect(&format!("Variable not found: {id}")),
        NumLiteral(n) => Value::U256(*n),
        StrLiteral(s) => Value::Str(s.clone()),
        BoolLiteral(b) => Value::Bool(*b),
        FnInvoke(name, args) => {
            let mut arg_vals = vec![];
            for arg in args.iter() {
                arg_vals.push(eval(arg, frame)?);
            }

            if let Some(func) = frame.get_fn(*name) {
                func.call(&arg_vals, frame)
            } else {
                panic!("Unknown function {name:?}");
            }
        }
        Add(lhs, rhs) => eval(lhs, frame)? + eval(rhs, frame)?,
        Sub(lhs, rhs) => eval(lhs, frame)? - eval(rhs, frame)?,
        Mul(lhs, rhs) => eval(lhs, frame)? * eval(rhs, frame)?,
        Div(lhs, rhs) => eval(lhs, frame)? / eval(rhs, frame)?,
        Gt(lhs, rhs) => {
            if eval(lhs, frame)? > eval(rhs, frame)? {
                Value::Bool(true)
            } else {
                Value::Bool(false)
            }
        }
        Lt(lhs, rhs) => {
            if eval(lhs, frame)? < eval(rhs, frame)? {
                Value::Bool(true)
            } else {
                Value::Bool(false)
            }
        }
        If(cond, t_case, f_case) => match eval(cond, frame)? {
            Value::Bool(true) => eval_stmts(t_case, frame)?,
            Value::Bool(false) => {
                if let Some(f_case) = f_case {
                    eval_stmts(f_case, frame)?
                } else {
                    Value::U256(U256::zero())
                }
            }
            _ => panic!("If condition must be a boolean"),
        },
    };
    EvalResult::Continue(res)
}

fn space_delimited<'src, O, E>(
    f: impl Parser<&'src str, O, E>,
) -> impl FnMut(&'src str) -> IResult<&'src str, O, E>
where
    E: ParseError<&'src str>,
{
    delimited(multispace0, f, multispace0)
}

fn factor(i: &str) -> IResult<&str, Expression> {
    alt((str_literal, num_literal, func_call, ident, parens))(i)
}

fn func_call(i: &str) -> IResult<&str, Expression> {
    let (r, ident) = space_delimited(identifier)(i)?;
    let (r, args) = space_delimited(delimited(
        tag("("),
        many0(delimited(multispace0, expr, space_delimited(opt(tag(","))))),
        tag(")"),
    ))(r)?;
    Ok((r, Expression::FnInvoke(ident, args)))
}

fn ident(input: &str) -> IResult<&str, Expression> {
    let (r, res) = space_delimited(identifier)(input)?;
    Ok((r, Expression::Ident(res)))
}

fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn str_literal(i: &str) -> IResult<&str, Expression> {
    let (r0, _) = preceded(multispace0, char('\"'))(i)?;
    let (r, val) = many0(none_of("\""))(r0)?;
    let (r, _) = terminated(char('"'), multispace0)(r)?;
    Ok((
        r,
        Expression::StrLiteral(
            val.iter()
                .collect::<String>()
                .replace("\\\\", "\\")
                .replace("\\n", "\n"),
        ),
    ))
}

fn num_literal(input: &str) -> IResult<&str, Expression> {
    let (r, v) = space_delimited(recognize_float)(input)?;
    Ok((
        r,
        Expression::NumLiteral(v.parse().map_err(|_| {
            nom::Err::Error(nom::error::Error {
                input,
                code: nom::error::ErrorKind::Digit,
            })
        })?),
    ))
}

fn parens(i: &str) -> IResult<&str, Expression> {
    space_delimited(delimited(tag("("), expr, tag(")")))(i)
}

fn term(i: &str) -> IResult<&str, Expression> {
    let (i, init) = factor(i)?;

    fold_many0(
        pair(space_delimited(alt((char('*'), char('/')))), factor),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| match op {
            '*' => Expression::Mul(Box::new(acc), Box::new(val)),
            '/' => Expression::Div(Box::new(acc), Box::new(val)),
            _ => panic!("Multiplicative expression should have '*' or '/' operator"),
        },
    )(i)
}

fn num_expr(i: &str) -> IResult<&str, Expression> {
    let (i, init) = term(i)?;

    fold_many0(
        pair(space_delimited(alt((char('+'), char('-')))), term),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| match op {
            '+' => Expression::Add(Box::new(acc), Box::new(val)),
            '-' => Expression::Sub(Box::new(acc), Box::new(val)),
            _ => {
                panic!("Additive expression should have '+' or '-' operator")
            }
        },
    )(i)
}

fn cond_expr(i: &str) -> IResult<&str, Expression> {
    let (i, first) = num_expr(i)?;
    let (i, cond) = space_delimited(alt((char('<'), char('>'))))(i)?;
    let (i, second) = num_expr(i)?;
    Ok((
        i,
        match cond {
            '<' => Expression::Lt(Box::new(first), Box::new(second)),
            '>' => Expression::Gt(Box::new(first), Box::new(second)),
            _ => unreachable!(),
        },
    ))
}

fn open_brace(i: &str) -> IResult<&str, ()> {
    let (i, _) = space_delimited(char('{'))(i)?;
    Ok((i, ()))
}

fn close_brace(i: &str) -> IResult<&str, ()> {
    let (i, _) = space_delimited(char('}'))(i)?;
    Ok((i, ()))
}

fn if_expr(i: &str) -> IResult<&str, Expression> {
    let (i, _) = space_delimited(tag("if"))(i)?;
    let (i, cond) = parens(i)?;
    let (i, t_case) = delimited(open_brace, statements, close_brace)(i)?;
    let (i, f_case) = opt(preceded(
        space_delimited(tag("else")),
        alt((
            delimited(open_brace, statements, close_brace),
            map_res(
                if_expr,
                |v| -> Result<Vec<Statement>, nom::error::Error<&str>> {
                    Ok(vec![Statement::Expression(v)])
                },
            ),
        )),
    ))(i)?;

    Ok((
        i,
        Expression::If(Box::new(cond), Box::new(t_case), f_case.map(Box::new)),
    ))
}
fn expr(i: &str) -> IResult<&str, Expression> {
    alt((if_expr, cond_expr, num_expr))(i)
}

fn var_def(i: &str) -> IResult<&str, Statement> {
    let (i, _) = delimited(multispace0, tag("var"), multispace1)(i)?;
    let (i, name) = space_delimited(identifier)(i)?;
    let (i, _) = space_delimited(char('='))(i)?;
    let (i, expr) = space_delimited(expr)(i)?;
    Ok((i, Statement::VarDef(name, expr)))
}

fn var_assign(i: &str) -> IResult<&str, Statement> {
    let (i, name) = space_delimited(identifier)(i)?;
    let (i, _) = space_delimited(char('='))(i)?;
    let (i, expr) = space_delimited(expr)(i)?;
    Ok((i, Statement::VarAssign(name, expr)))
}

fn expr_statement(i: &str) -> IResult<&str, Statement> {
    let (i, res) = expr(i)?;
    Ok((i, Statement::Expression(res)))
}

fn fn_def_statement(i: &str) -> IResult<&str, Statement> {
    let (i, _) = space_delimited(tag("function"))(i)?;
    let (i, name) = space_delimited(identifier)(i)?;
    let (i, _) = space_delimited(tag("("))(i)?;
    let (i, args) = separated_list0(char(','), space_delimited(identifier))(i)?;
    let (i, _) = space_delimited(tag(")"))(i)?;
    let (i, stmts) = delimited(open_brace, statements, close_brace)(i)?;
    Ok((i, Statement::FnDef { name, args, stmts }))
}

fn return_statement(i: &str) -> IResult<&str, Statement> {
    let (i, _) = space_delimited(tag("return"))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    Ok((i, Statement::Return(ex)))
}

fn general_statement<'a>() -> impl Fn(&'a str) -> IResult<&'a str, Statement> {
    let terminator = move |i| -> IResult<&str, ()> {
        let mut semicolon = pair(tag(";"), multispace0);
        Ok((semicolon(i)?.0, ()))
    };
    move |input| {
        alt((
            terminated(var_def, terminator),
            terminated(var_assign, terminator),
            fn_def_statement,
            terminated(return_statement, terminator),
            terminated(expr_statement, terminator),
        ))(input)
    }
}

pub(crate) fn statement(input: &str) -> IResult<&str, Statement> {
    general_statement()(input)
}

fn statements(i: &str) -> IResult<&str, Statements> {
    let (i, mut stmts) = many0(statement)(i)?;
    let (i, _) = opt(multispace0)(i)?;

    Ok((i, stmts))
}

fn statements_finish(i: &str) -> Result<Statements, nom::error::Error<&str>> {
    let (_, res) = statements(i).finish()?;
    Ok(res)
}
