use primitive_types::U256;

use std::io::Read;
use std::{cmp::Ordering, collections::HashMap, env, fs::File, ops::ControlFlow};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1, none_of},
    combinator::{cut, map_res, opt, recognize},
    error::ParseError,
    multi::{fold_many0, many0, separated_list0},
    number::complete::recognize_float,
    sequence::{delimited, pair, preceded, terminated},
    Finish, IResult, InputTake, Offset, Parser,
};
use nom_locate::LocatedSpan;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("no file name provided");
        std::process::exit(1);
    }

    let file_name = &args[1];

    let mut src_file = File::open(file_name)?;

    let mut source = String::new();

    src_file.read_to_string(&mut source)?;

    let parsed_statements = match statements_finish(Span::new(&source)) {
        Ok(stmts) => stmts,
        Err(err) => {
            eprintln!(
                "Parse error: {}:{}: {err}",
                err.input.location_line(),
                err.input.get_column()
            );
            return Ok(());
        }
    };

    println!("AST: {parsed_statements:#?}");

    let mut tc_ctx = TypeCheckContext::new();

    if let Err(err) = type_check(&parsed_statements, &mut tc_ctx) {
        println!(
            "Type check error: {}:{}: {err}",
            err.span.location_line(),
            err.span.get_column()
        );
        return Ok(());
    }
    println!("Type check OK");

    let mut frame = StackFrame::new();

    eval_stmts(&parsed_statements, &mut frame);

    Ok(())
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

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TypeDecl {
    Str,
    Bool,
    U256,
}

fn tc_check_type<'src>(
    value: &Option<TypeDecl>,
    target: &Option<TypeDecl>,
    span: Span<'src>,
) -> Result<Option<TypeDecl>, TypeCheckError<'src>> {
    if value == target {
        Ok(*value)
    } else {
        Err(TypeCheckError::new(
            format!("{:?} cannot be assigned to {:?}", value, target),
            span,
        ))
    }
}

pub struct TypeCheckContext<'src, 'ctx> {
    /// Variables table for type checking.
    vars: HashMap<&'src str, TypeDecl>,
    /// Function names are owned strings because it can be either from source or native.
    funcs: HashMap<String, FnDef<'src>>,
    super_context: Option<&'ctx TypeCheckContext<'src, 'ctx>>,
}

impl<'src, 'ctx> TypeCheckContext<'src, 'ctx> {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            funcs: standard_functions(),
            super_context: None,
        }
    }

    fn get_var(&self, name: &str) -> Option<TypeDecl> {
        if let Some(val) = self.vars.get(name) {
            Some(val.clone())
        } else {
            None
        }
    }

    fn get_fn(&self, name: &str) -> Option<&FnDef<'src>> {
        if let Some(val) = self.funcs.get(name) {
            Some(val)
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_fn(name)
        } else {
            None
        }
    }

    fn push_stack(super_ctx: &'ctx Self) -> Self {
        Self {
            vars: HashMap::new(),
            funcs: HashMap::new(),
            super_context: Some(super_ctx),
        }
    }
}

#[derive(Debug)]
pub struct TypeCheckError<'src> {
    msg: String,
    span: Span<'src>,
}

impl<'src> std::fmt::Display for TypeCheckError<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\nlocation: {}:{}: {}",
            self.msg,
            self.span.location_line(),
            self.span.get_utf8_column(),
            self.span.fragment()
        )
    }
}

impl<'src> TypeCheckError<'src> {
    fn new(msg: String, span: Span<'src>) -> Self {
        Self { msg, span }
    }
}

fn tc_binary_op<'src>(
    lhs: &Expression<'src>,
    rhs: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
    op: &str,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    let lhst = tc_expr(lhs, ctx)?;
    let rhst = tc_expr(rhs, ctx)?;
    binary_op_type(&lhst, &rhst).map_err(|_| {
        TypeCheckError::new(
            format!(
                "Operation {op} between incompatible type: {:?} and {:?}",
                lhst, rhst,
            ),
            lhs.span,
        )
    })
}

fn binary_op_type(lhs: &Option<TypeDecl>, rhs: &Option<TypeDecl>) -> Result<TypeDecl, ()> {
    use TypeDecl::*;
    Ok(match (lhs, rhs) {
        (Some(U256), Some(U256)) => U256,
        _ => return Err(()),
    })
}

fn tc_binary_cmp<'src>(
    lhs: &Expression<'src>,
    rhs: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
    op: &str,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    use TypeDecl::*;
    let lhst = tc_expr(lhs, ctx)?;
    let rhst = tc_expr(rhs, ctx)?;
    Ok(match (&lhst, &rhst) {
        (Some(U256), Some(U256)) => Bool,
        _ => {
            return Err(TypeCheckError::new(
                format!(
                    "Operation {op} between incompatible type: {:?} and {:?}",
                    lhst, rhst,
                ),
                lhs.span,
            ))
        }
    })
}

fn tc_expr<'src>(
    e: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
) -> Result<Option<TypeDecl>, TypeCheckError<'src>> {
    use ExprEnum::*;
    Ok(match &e.expr {
        NumLiteral(_val) => Some(TypeDecl::U256),
        StrLiteral(_val) => Some(TypeDecl::Str),
        BoolLiteral(_val) => Some(TypeDecl::Bool),
        Ident(str) => {
            let v = ctx.get_var(str);
            if v == None {
                Err(TypeCheckError::new(
                    format!("Variable \"{}\" not found in scope", str),
                    e.span,
                ))?;
            }
            v
        }
        FnInvoke(str, args) => {
            let args_ty = args
                .iter()
                .map(|v| Ok((tc_expr(v, ctx)?, v.span)))
                .collect::<Result<Vec<_>, _>>()?;
            let func = ctx.get_fn(**str).ok_or_else(|| {
                TypeCheckError::new(format!("function {} is not defined", str), *str)
            })?;
            let args_decl = func.args();
            for ((arg_ty, arg_span), decl) in args_ty.iter().zip(args_decl.iter()) {
                tc_check_type(&arg_ty, &Some(decl.1), *arg_span)?;
            }
            func.ret_type()
        }
        Add(lhs, rhs) => Some(tc_binary_op(&lhs, &rhs, ctx, "Add")?),
        Sub(lhs, rhs) => Some(tc_binary_op(&lhs, &rhs, ctx, "Sub")?),
        Mul(lhs, rhs) => Some(tc_binary_op(&lhs, &rhs, ctx, "Mult")?),
        Div(lhs, rhs) => Some(tc_binary_op(&lhs, &rhs, ctx, "Div")?),
        Lt(lhs, rhs) => Some(tc_binary_cmp(&lhs, &rhs, ctx, "LT")?),
        Gt(lhs, rhs) => Some(tc_binary_cmp(&lhs, &rhs, ctx, "GT")?),
        If(cond, true_branch, false_branch) => {
            let t = tc_check_type(&tc_expr(cond, ctx)?, &Some(TypeDecl::Bool), cond.span)?;
            // todo: may be nothing to return
            t
        }
    })
}

fn type_check<'src>(
    stmts: &Vec<Statement<'src>>,
    ctx: &mut TypeCheckContext<'src, '_>,
) -> Result<Option<TypeDecl>, TypeCheckError<'src>> {
    let mut res = None;
    for stmt in stmts {
        match stmt {
            Statement::VarDef { name, td, ex, .. } => {
                let init_type = tc_expr(ex, ctx)?;
                let init_type = tc_check_type(&init_type, &Some(*td), ex.span)?;

                if let Some(var_type) = init_type {
                    ctx.vars.insert(**name, var_type);
                } else {
                    return Err(TypeCheckError::new("Must have type".to_string(), ex.span));
                }
            }
            Statement::VarAssign { name, ex, .. } => {
                let init_type = tc_expr(ex, ctx)?;
                let target = ctx.vars.get(**name).expect("Variable not found");
                tc_check_type(&init_type, &Some(*target), ex.span)?;
            }
            Statement::FnDef {
                name,
                args,
                ret_type,
                stmts,
            } => {
                // Function declaration needs to be added first to allow recursive calls
                ctx.funcs.insert(
                    name.to_string(),
                    FnDef::User(UserFn {
                        args: args.clone(),
                        ret_type: *ret_type,
                        stmts: stmts.clone(),
                    }),
                );
                let mut subctx = TypeCheckContext::push_stack(ctx);
                for (arg, ty) in args.iter() {
                    subctx.vars.insert(arg, *ty);
                }
                let last_stmt = type_check(stmts, &mut subctx)?;
                tc_check_type(&last_stmt, ret_type, stmts.span())?;
            }
            Statement::Expression(e) => {
                res = tc_expr(&e, ctx)?;
            }
            Statement::Return(e) => {
                return tc_expr(e, ctx);
            }
        }
    }
    Ok(res)
}

enum FnDef<'src> {
    User(UserFn<'src>),
    Native(NativeFn<'src>),
}

impl<'src> FnDef<'src> {
    fn call(&self, args: &[Value], frame: &StackFrame) -> Value {
        match self {
            Self::User(code) => {
                let mut new_frame = StackFrame::push_stack(frame);
                new_frame.vars = args
                    .iter()
                    .zip(code.args.iter())
                    .map(|(arg, decl)| (decl.0.to_string(), arg.clone()))
                    .collect();
                match eval_stmts(&code.stmts, &mut new_frame) {
                    EvalResult::Continue(val) | EvalResult::Break(BreakResult::Return(val)) => val,
                    EvalResult::Break(BreakResult::Break) => {
                        panic!("Breaking outside loop is prohibited")
                    }
                    EvalResult::Break(BreakResult::Continue) => {
                        panic!("Continuing outside loop is prohibited")
                    }
                }
            }
            Self::Native(code) => (code.code)(args),
        }
    }

    fn args(&self) -> Vec<(&'src str, TypeDecl)> {
        match self {
            Self::User(user) => user.args.iter().map(|arg| (&**arg.0, arg.1)).collect(),
            Self::Native(code) => code.args.clone(),
        }
    }

    fn ret_type(&self) -> Option<TypeDecl> {
        match self {
            Self::User(user) => user.ret_type,
            Self::Native(native) => native.ret_type,
        }
    }
}

struct UserFn<'src> {
    args: Vec<(Span<'src>, TypeDecl)>,
    ret_type: Option<TypeDecl>,
    stmts: Statements<'src>,
}

struct NativeFn<'src> {
    args: Vec<(&'src str, TypeDecl)>,
    ret_type: Option<TypeDecl>,
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
        Self {
            vars: Variables::new(),
            funcs: standard_functions(),
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

fn standard_functions<'src>() -> Functions<'src> {
    let mut funcs = Functions::new();
    // funcs.insert("sqrt".to_string(), unary_fn(f64::sqrt));

    funcs
}

fn eval_stmts<'src>(stmts: &[Statement<'src>], frame: &mut StackFrame<'src>) -> EvalResult {
    let mut last_result = EvalResult::Continue(Value::U256(U256::zero()));
    for statement in stmts {
        match statement {
            Statement::Expression(expr) => {
                last_result = EvalResult::Continue(eval(expr, frame)?);
            }
            Statement::VarDef { name, ex, .. } => {
                let value = eval(ex, frame)?;
                frame.vars.insert(name.to_string(), value);
            }
            Statement::VarAssign { name, ex, .. } => {
                if !frame.vars.contains_key(**name) {
                    panic!("Variable is not defined");
                }
                let value = eval(ex, frame)?;
                frame.vars.insert(name.to_string(), value);
            }
            Statement::FnDef {
                name,
                args,
                ret_type,
                stmts,
            } => {
                frame.funcs.insert(
                    name.to_string(),
                    FnDef::User(UserFn {
                        args: args.clone(),
                        ret_type: *ret_type,
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
enum ExprEnum<'src> {
    Ident(Span<'src>),
    NumLiteral(U256),
    StrLiteral(String),
    BoolLiteral(bool),
    FnInvoke(Span<'src>, Vec<Expression<'src>>),
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
struct Expression<'a> {
    pub(crate) expr: ExprEnum<'a>,
    pub(crate) span: Span<'a>,
}

impl<'a> Expression<'a> {
    fn new(expr: ExprEnum<'a>, span: Span<'a>) -> Self {
        Self { expr, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum Statement<'src> {
    Expression(Expression<'src>),
    VarDef {
        span: Span<'src>,
        name: Span<'src>,
        td: TypeDecl,
        ex: Expression<'src>,
    },
    VarAssign {
        span: Span<'src>,
        name: Span<'src>,
        ex: Expression<'src>,
    },
    FnDef {
        name: Span<'src>,
        args: Vec<(Span<'src>, TypeDecl)>,
        ret_type: Option<TypeDecl>,
        stmts: Statements<'src>,
    },
    Return(Expression<'src>),
}

impl<'src> Statement<'src> {
    fn span(&self) -> Option<Span<'src>> {
        use Statement::*;
        Some(match self {
            Expression(ex) => ex.span,
            VarDef { span, .. } => *span,
            VarAssign { span, .. } => *span,
            FnDef { name, stmts, .. } => calc_offset(*name, stmts.span()),
            Return(ex) => ex.span,
        })
    }
}

trait GetSpan<'a> {
    fn span(&self) -> Span<'a>;
}

type Statements<'a> = Vec<Statement<'a>>;

impl<'a> GetSpan<'a> for Statements<'a> {
    fn span(&self) -> Span<'a> {
        self.iter().find_map(|stmt| stmt.span()).unwrap()
    }
}

// fn unary_fn<'a>(f: fn(f64) -> f64) -> FnDef<'a> {
//     FnDef::Native(NativeFn {
//         args: vec![("lhs", TypeDecl::F64), ("rhs", TypeDecl::F64)],
//         ret_type: TypeDecl::F64,
//         code: Box::new(move |args| {
//             Value::F64(f(coerce_f64(
//                 args.into_iter().next().expect("function missing argument"),
//             )))
//         }),
//     })
// }

// fn binary_fn<'a>(f: fn(f64, f64) -> f64) -> FnDef<'a> {
//     FnDef::Native(NativeFn {
//         args: vec![("lhs", TypeDecl::F64), ("rhs", TypeDecl::F64)],
//         ret_type: TypeDecl::F64,
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
    Return(Option<Value>),
    // Break,
    // Continue,
}

type EvalResult = ControlFlow<BreakResult, Option<Value>>;

fn eval<'src>(expr: &Expression<'src>, frame: &mut StackFrame<'src>) -> EvalResult {
    use ExprEnum::*;
    let res = match &expr.expr {
        Ident(id) => frame
            .vars
            .get(**id)
            .cloned()
            .expect(&format!("Variable {id:?} not found")),
        NumLiteral(n) => Value::U256(*n),
        StrLiteral(s) => Value::Str(s.clone()),
        FnInvoke(name, args) => {
            let mut arg_vals = vec![];
            for arg in args.iter() {
                arg_vals.push(eval(arg, frame)?);
            }

            if let Some(func) = frame.get_fn(**name) {
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
        If(cond, t_case, f_case) => {
            if eval(cond, frame)? != Value::Bool(false) {
                eval_stmts(t_case, frame)?
            } else if let Some(f_case) = f_case {
                eval_stmts(f_case, frame)?
            } else {
                None
            }
        }
    };
    EvalResult::Continue(res)
}

fn space_delimited<'src, O, E>(
    f: impl Parser<Span<'src>, O, E>,
) -> impl FnMut(Span<'src>) -> IResult<Span<'src>, O, E>
where
    E: ParseError<Span<'src>>,
{
    delimited(multispace0, f, multispace0)
}

/// Calculate offset between the start positions of the input spans and return a span between them.
///
/// Note: `i` shall start earlier than `r`, otherwise wrapping would occur.
fn calc_offset<'a>(i: Span<'a>, r: Span<'a>) -> Span<'a> {
    i.take(i.offset(&r))
}

fn factor(i: Span) -> IResult<Span, Expression> {
    alt((str_literal, num_literal, func_call, ident, parens))(i)
}

fn func_call(i: Span) -> IResult<Span, Expression> {
    let (r, ident) = space_delimited(identifier)(i)?;
    let (r, args) = space_delimited(delimited(
        tag("("),
        many0(delimited(multispace0, expr, space_delimited(opt(tag(","))))),
        tag(")"),
    ))(r)?;
    Ok((
        r,
        Expression {
            expr: ExprEnum::FnInvoke(ident, args),
            span: i,
        },
    ))
}

fn ident(input: Span) -> IResult<Span, Expression> {
    let (r, res) = space_delimited(identifier)(input)?;
    Ok((
        r,
        Expression {
            expr: ExprEnum::Ident(res),
            span: input,
        },
    ))
}

fn identifier(input: Span) -> IResult<Span, Span> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn str_literal(i: Span) -> IResult<Span, Expression> {
    let (r0, _) = preceded(multispace0, char('\"'))(i)?;
    let (r, val) = many0(none_of("\""))(r0)?;
    let (r, _) = terminated(char('"'), multispace0)(r)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::StrLiteral(
                val.iter()
                    .collect::<String>()
                    .replace("\\\\", "\\")
                    .replace("\\n", "\n"),
            ),
            i,
        ),
    ))
}

fn num_literal(input: Span) -> IResult<Span, Expression> {
    let (r, v) = space_delimited(recognize_float)(input)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::NumLiteral(v.parse().map_err(|_| {
                nom::Err::Error(nom::error::Error {
                    input,
                    code: nom::error::ErrorKind::Digit,
                })
            })?),
            v,
        ),
    ))
}

fn parens(i: Span) -> IResult<Span, Expression> {
    space_delimited(delimited(tag("("), expr, tag(")")))(i)
}

fn term(i: Span) -> IResult<Span, Expression> {
    let (r, init) = factor(i)?;

    let res = fold_many0(
        pair(space_delimited(alt((char('*'), char('/')))), factor),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| {
            let span = calc_offset(i, acc.span);
            match op {
                '*' => Expression::new(ExprEnum::Mul(Box::new(acc), Box::new(val)), span),
                '/' => Expression::new(ExprEnum::Div(Box::new(acc), Box::new(val)), span),
                _ => panic!(
                    "Multiplicative expression should have '*' \
            or '/' operator"
                ),
            }
        },
    )(r);
    res
}

fn num_expr(i: Span) -> IResult<Span, Expression> {
    let (r, init) = term(i)?;

    let res = fold_many0(
        pair(space_delimited(alt((char('+'), char('-')))), term),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| {
            let span = calc_offset(i, acc.span);
            match op {
                '+' => Expression::new(ExprEnum::Add(Box::new(acc), Box::new(val)), span),
                '-' => Expression::new(ExprEnum::Sub(Box::new(acc), Box::new(val)), span),
                _ => panic!("Additive expression should have '+' or '-' operator"),
            }
        },
    )(r);
    res
}

fn cond_expr(i0: Span) -> IResult<Span, Expression> {
    let (i, first) = num_expr(i0)?;
    let (i, cond) = space_delimited(alt((char('<'), char('>'))))(i)?;
    let (i, second) = num_expr(i)?;
    let span = calc_offset(i0, i);
    Ok((
        i,
        match cond {
            '<' => Expression::new(ExprEnum::Lt(Box::new(first), Box::new(second)), span),
            '>' => Expression::new(ExprEnum::Gt(Box::new(first), Box::new(second)), span),
            _ => unreachable!(),
        },
    ))
}

fn open_brace(i: Span) -> IResult<Span, ()> {
    let (i, _) = space_delimited(char('{'))(i)?;
    Ok((i, ()))
}

fn close_brace(i: Span) -> IResult<Span, ()> {
    let (i, _) = space_delimited(char('}'))(i)?;
    Ok((i, ()))
}

fn if_expr(i0: Span) -> IResult<Span, Expression> {
    let (i, _) = space_delimited(tag("if"))(i0)?;
    let (i, cond) = expr(i)?;
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
        Expression::new(
            ExprEnum::If(Box::new(cond), Box::new(t_case), f_case.map(Box::new)),
            calc_offset(i0, i),
        ),
    ))
}

fn expr(i: Span) -> IResult<Span, Expression> {
    alt((if_expr, cond_expr, num_expr))(i)
}

fn var_def(i: Span) -> IResult<Span, Statement> {
    let span = i;
    let (i, _) = delimited(multispace0, tag("var"), multispace1)(i)?;
    let (i, (name, td, ex)) = cut(|i| {
        let (i, name) = space_delimited(identifier)(i)?;
        let (i, _) = space_delimited(char(':'))(i)?;
        let (i, td) = type_decl(i)?;
        let (i, _) = space_delimited(char('='))(i)?;
        let (i, ex) = space_delimited(expr)(i)?;
        let (i, _) = space_delimited(char(';'))(i)?;
        Ok((i, (name, td, ex)))
    })(i)?;
    Ok((
        i,
        Statement::VarDef {
            span: calc_offset(span, i),
            name,
            td,
            ex,
        },
    ))
}

fn var_assign(i: Span) -> IResult<Span, Statement> {
    let span = i;
    let (i, name) = space_delimited(identifier)(i)?;
    let (i, _) = space_delimited(char('='))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    let (i, _) = space_delimited(char(';'))(i)?;
    Ok((
        i,
        Statement::VarAssign {
            span: calc_offset(span, i),
            name,
            ex,
        },
    ))
}

fn expr_statement(i: Span) -> IResult<Span, Statement> {
    let (i, res) = expr(i)?;
    Ok((i, Statement::Expression(res)))
}

fn for_statement(i: Span) -> IResult<Span, Statement> {
    let i0 = i;
    let (i, _) = space_delimited(tag("for"))(i)?;
    let (i, (loop_var, start, end, stmts)) = cut(|i| {
        let (i, loop_var) = space_delimited(identifier)(i)?;
        let (i, _) = space_delimited(tag("in"))(i)?;
        let (i, start) = space_delimited(expr)(i)?;
        let (i, _) = space_delimited(tag("to"))(i)?;
        let (i, end) = space_delimited(expr)(i)?;
        let (i, stmts) = delimited(open_brace, statements, close_brace)(i)?;
        Ok((i, (loop_var, start, end, stmts)))
    })(i)?;
    Ok((
        i,
        Statement::For {
            span: calc_offset(i0, i),
            loop_var,
            start,
            end,
            stmts,
        },
    ))
}

fn type_decl(i: Span) -> IResult<Span, TypeDecl> {
    let (i, td) = space_delimited(identifier)(i)?;
    Ok((
        i,
        match *td.fragment() {
            "i64" => TypeDecl::I64,
            "f64" => TypeDecl::F64,
            "str" => TypeDecl::Str,
            _ => {
                return Err(nom::Err::Failure(nom::error::Error::new(
                    td,
                    nom::error::ErrorKind::Verify,
                )));
            }
        },
    ))
}

fn argument(i: Span) -> IResult<Span, (Span, TypeDecl)> {
    let (i, ident) = space_delimited(identifier)(i)?;
    let (i, _) = char(':')(i)?;
    let (i, td) = type_decl(i)?;

    Ok((i, (ident, td)))
}

fn fn_def_statement(i: Span) -> IResult<Span, Statement> {
    let (i, _) = space_delimited(tag("fn"))(i)?;
    let (i, (name, args, ret_type, stmts)) = cut(|i| {
        let (i, name) = space_delimited(identifier)(i)?;
        let (i, _) = space_delimited(tag("("))(i)?;
        let (i, args) = separated_list0(char(','), space_delimited(argument))(i)?;
        let (i, _) = space_delimited(tag(")"))(i)?;
        let (i, _) = space_delimited(tag("->"))(i)?;
        let (i, ret_type) = type_decl(i)?;
        let (i, stmts) = delimited(open_brace, statements, close_brace)(i)?;
        Ok((i, (name, args, ret_type, stmts)))
    })(i)?;
    Ok((
        i,
        Statement::FnDef {
            name,
            args,
            ret_type,
            stmts,
        },
    ))
}

fn return_statement(i: Span) -> IResult<Span, Statement> {
    let (i, _) = space_delimited(tag("return"))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    Ok((i, Statement::Return(ex)))
}

fn general_statement<'a>(last: bool) -> impl Fn(Span<'a>) -> IResult<Span<'a>, Statement> {
    let terminator = move |i| -> IResult<Span, ()> {
        let mut semicolon = pair(tag(";"), multispace0);
        if last {
            Ok((opt(semicolon)(i)?.0, ()))
        } else {
            Ok((semicolon(i)?.0, ()))
        }
    };
    move |input| {
        alt((
            var_def,
            var_assign,
            fn_def_statement,
            for_statement,
            terminated(return_statement, terminator),
            terminated(expr_statement, terminator),
        ))(input)
    }
}

pub(crate) fn last_statement(input: Span) -> IResult<Span, Statement> {
    general_statement(true)(input)
}

pub(crate) fn statement(input: Span) -> IResult<Span, Statement> {
    general_statement(false)(input)
}

fn statements(i: Span) -> IResult<Span, Statements> {
    let (i, mut stmts) = many0(statement)(i)?;
    let (i, last) = opt(last_statement)(i)?;
    let (i, _) = opt(multispace0)(i)?;
    if let Some(last) = last {
        stmts.push(last);
    }
    Ok((i, stmts))
}

fn statements_finish(i: Span) -> Result<Statements, nom::error::Error<Span>> {
    let (_, res) = statements(i).finish()?;
    Ok(res)
}
