import Lean

open Lean Meta Elab Command

private def natLeArgs? (e : Expr) : Option (Expr × Expr) := do
  let e := e.consumeMData
  let args := e.getAppArgs
  if e.isAppOfArity ``LE.le 4 && args[0]!.isConstOf ``Nat then
    return (args[2]!, args[3]!)
  if e.isAppOfArity ``Nat.le 2 then
    return (args[0]!, args[1]!)
  none

private partial def encodeExpr (e : Expr) : MetaM Json := do
  let e := e.consumeMData
  let args := e.getAppArgs
  if e.isAppOfArity ``HAdd.hAdd 6 && (args.extract 0 3).all (·.isConstOf ``Nat) then
    return .arr #[.str "add", ← encodeExpr args[4]!, ← encodeExpr args[5]!]
  if e.isAppOfArity ``Nat.add 2 then
    return .arr #[.str "add", ← encodeExpr args[0]!, ← encodeExpr args[1]!]
  if e.isFVar then
    return .arr #[.str "var", .str (← e.fvarId!.getDecl).userName.toString]
  if let some n ← getNatValue? e then
    return .arr #[.str "nat", toJson n]
  throwError "unsupported natural-number expression: {e}"

elab "inspect_goal_tree " label:ident : command => do
  let name := label.getId
  let type := (← getConstInfo name).type
  liftTermElabM <| forallTelescope type fun binders conclusion => do
    let some (left, right) := natLeArgs? conclusion
      | throwError "expected a natural-number inequality"
    let mut assumptions := #[]
    for binder in binders do
      let decl ← binder.fvarId!.getDecl
      if let some (a, b) := natLeArgs? decl.type then
        assumptions := assumptions.push <| .arr #[.str decl.userName.toString,
          ← encodeExpr a, ← encodeExpr b]
    let tree := Json.mkObj [("left", ← encodeExpr left), ("right", ← encodeExpr right),
      ("assumptions", .arr assumptions)]
    logInfo m!"GOAL_TREE {name} {tree.compress}"
