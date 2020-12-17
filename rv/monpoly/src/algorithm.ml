(*
 * This file is part of MONPOLY.
 *
 * Copyright (C) 2011 Nokia Corporation and/or its subsidiary(-ies).
 * Contact:  Nokia Corporation (Debmalya Biswas: debmalya.biswas@nokia.com)
 *
 * Copyright (C) 2012 ETH Zurich.
 * Contact:  ETH Zurich (Eugen Zalinescu: eugen.zalinescu@inf.ethz.ch)
 *
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see
 * http://www.gnu.org/licenses/lgpl-2.1.html.
 *
 * As a special exception to the GNU Lesser General Public License,
 * you may link, statically or dynamically, a "work that uses the
 * Library" with a publicly distributed version of the Library to
 * produce an executable file containing portions of the Library, and
 * distribute that executable file under terms of your choice, without
 * any of the additional requirements listed in clause 6 of the GNU
 * Lesser General Public License. By "a publicly distributed version
 * of the Library", we mean either the unmodified Library as
 * distributed by Nokia, or a modified version of the Library that is
 * distributed under the conditions defined in clause 3 of the GNU
 * Lesser General Public License. This exception does not however
 * invalidate any other reasons why the executable file might be
 * covered by the GNU Lesser General Public License.
 *)



(** This module implements the monitoring algorithm. This algorithm is
    described in the paper "Runtime Monitoring of Metric First-order
    Temporal Properties" by David Basin, Felix Klaedtke, Samuel
    Muller, and Birgit Pfitzmann, presented at FSTTCS'08.


    This is the MONPOLY's main module, all other modules can be seen
    as "helper" modules. The module's entry point is normally the
    [monitor] function. This function checks that the given formula is
    monitorable and then calls the [check_log] function which
    iteratively reads each log entry. To be able to incrementally
    process the entries, the input formula is first extended with
    additional information for each subformula, by calling the
    [add_ext] function.  Also, a queue [neval] of not-yet evaluated
    indexes of log entries is maintained.

    The function [check_log] reads each log entry, calls [add_index]
    to update the extended formula with the new information from the
    entry at index [i], adds index [i] to the queue of not-yet
    evaluated indexes, and finally calls [process_index] to process
    this entry.

    The function [process_index] iterativelly tries to evaluate the
    formula at each index (calling the function [eval]) from the queue
    of not-yet evaluated indexes. It stops when the formula cannot be
    evaluated or when the formula has been evaluated at all indexes in
    the queue. The function [eval] performs a bottom-up evaluation of
    the formula.
*)


open Dllist
open Misc
open Perf
open Predicate
open MFOTL
open Tuple
open Relation
open Table
open Db
open Log
open Sliding

module NEval = Dllist
module Sk = Dllist
module Sj = Dllist


(* For the sake of clarity, think about merging these types and all
   related functions. Some fields will be redundant, but we will not lose
   that much. *)

type info = (int * timestamp * relation) Queue.t
type ainfo = {mutable arel: relation option}
type pinfo = {mutable ptsq: timestamp}
type ninfo = {mutable init: bool}
type oainfo = {mutable ores: relation;
         oaauxrels: (timestamp * relation) Mqueue.t}

module IntMap = Map.Make (
  struct type t = cst
   let compare = Pervasives.compare
  end)

type t_agg =
  | CSA_aux of int * cst
  | Med_aux of (int * (int IntMap.t))

type agg_once_state = {
  tw_rels: (timestamp * (tuple * tuple * cst) list) Queue.t;
  other_rels: (timestamp * relation) Queue.t;
  mutable mset: (tuple, int) Hashtbl.t;
  mutable hres: (tuple, t_agg) Hashtbl.t;
}

type aggMM_once_state = {
  non_tw_rels: (timestamp * relation) Queue.t;
  mutable tbl: (tuple, (timestamp * cst) Dllist.dllist) Hashtbl.t;
}

type ozinfo = {mutable oztree: (int, relation) Sliding.stree;
               mutable ozlast: (int * timestamp * relation) Dllist.cell;
               ozauxrels: (int * timestamp * relation) Dllist.dllist}
type oinfo = {mutable otree: (timestamp, relation) Sliding.stree;
              mutable olast: (timestamp * relation) Dllist.cell;
              oauxrels: (timestamp * relation) Dllist.dllist}
type sainfo = {mutable sres: relation;
               mutable sarel2: relation option;
               saauxrels: (timestamp * relation) Mqueue.t}
type sinfo = {mutable srel2: relation option;
              sauxrels: (timestamp * relation) Mqueue.t}
type ezinfo = {mutable ezlastev: (int * timestamp) NEval.cell;
               mutable eztree: (int, relation) Sliding.stree;
               mutable ezlast: (int * timestamp * relation) Dllist.cell;
               ezauxrels: (int * timestamp * relation) Dllist.dllist}
type einfo = {mutable elastev: (int * timestamp) NEval.cell;
              mutable etree: (timestamp, relation) Sliding.stree;
              mutable elast: (timestamp * relation) Dllist.cell;
              eauxrels: (timestamp * relation) Dllist.dllist}
type uinfo = {mutable ulast: (int * timestamp) NEval.cell;
              mutable ufirst: bool;
              mutable ures: relation;
              mutable urel2: relation option;
              raux: (int * timestamp * (int * relation) Sk.dllist) Sj.dllist;
              mutable saux: (int * relation) Sk.dllist}
type uninfo = {mutable last1: (int * timestamp) NEval.cell;
               mutable last2: (int * timestamp) NEval.cell;
               mutable listrel1: (int * timestamp * relation) Dllist.dllist;
               mutable listrel2: (int * timestamp * relation) Dllist.dllist}

module Tuple_map = Map.Make (
  struct type t = tuple
    let compare = Tuple.compare
  end)



type comp_one = relation -> relation
type comp_two = relation -> relation -> relation

type extformula =
  | ERel of relation
  | EPred of predicate * comp_one * info
  | ENeg of extformula
  | EAnd of comp_two * extformula * extformula * ainfo
  | EOr of comp_two * extformula * extformula * ainfo
  | EExists of comp_one * extformula
  | EAggreg of comp_one * extformula
  | EAggOnce of extformula * interval * agg_once_state *
                (agg_once_state -> (tuple * tuple * cst) list -> unit) *
                (agg_once_state -> relation -> (tuple * tuple * cst) list) *
                (agg_once_state -> relation)
  | EAggMMOnce of extformula * interval * aggMM_once_state *
                  (aggMM_once_state -> timestamp -> unit) *
                  (aggMM_once_state -> timestamp -> relation -> unit) *
                  (aggMM_once_state -> relation)
  | EPrev of interval * extformula * pinfo
  | ENext of interval * extformula * ninfo
  | ESinceA of comp_two * interval * extformula * extformula * sainfo
  | ESince of comp_two * interval * extformula * extformula * sinfo
  | EOnceA of interval * extformula * oainfo
  | EOnceZ of interval * extformula * ozinfo
  | EOnce of interval * extformula * oinfo
  | ENUntil of comp_two * interval * extformula * extformula * uninfo
  | EUntil of comp_two * interval * extformula * extformula * uinfo
  | EEventuallyZ of interval * extformula * ezinfo
  | EEventually of interval * extformula * einfo


let crt_ts = ref MFOTL.ts_invalid
let crt_tp = ref (-1)


let print_bool b =
  if b then
    print_string "true"
  else
    print_string "false"

let print_neval str neval =
  print_string str;
  Misc.print_dllist
    (fun (q,tsq) ->
       Printf.printf "(%d,%s)" q (MFOTL.string_of_ts tsq)
    ) neval;
  print_newline()


let print_ainf str ainf =
  print_string str;
  match ainf with
  | None -> print_string "None"
  | Some rel -> Relation.print_rel "" rel

let print_auxel =
  (fun (k,rel) ->
     Printf.printf "(%d->" k;
     Relation.print_rel "" rel;
     print_string ")"
  )
let print_sauxel =
  (fun (tsq,rel) ->
     Printf.printf "(%s," (MFOTL.string_of_ts tsq);
     Relation.print_rel "" rel;
     print_string ")"
  )

let print_rauxel (j,tsj,rrelsj) =
  Printf.printf "(j=%d,tsj=" j;
  MFOTL.print_ts tsj;
  print_string ",r=";
  Misc.print_dllist print_auxel rrelsj;
  print_string "),"


let print_aauxel (q,tsq,rel) =
  Printf.printf "(%d,%s," q (MFOTL.string_of_ts tsq);
  Relation.print_rel "" rel;
  print_string ")"

let print_inf inf =
  Misc.print_queue print_aauxel inf

let print_predinf str inf =
  print_string str;
  print_inf inf;
  print_newline()

let print_ozinf str inf =
  print_string str;
  if inf.ozlast == Dllist.void then
    print_string "ozlast = None; "
  else
    begin
      let (j,_,_) = Dllist.get_data inf.ozlast in
      Printf.printf "ozlast (index) = %d; " j
    end;
  Misc.print_dllist print_aauxel inf.ozauxrels;
  Sliding.print_stree
    string_of_int
    (Relation.print_rel " ztree = ")
    "; ozinf.ztree = "
    inf.oztree

let print_oinf str inf =
  print_string (str ^ "{");
  if inf.olast == Dllist.void then
    print_string "last = None; "
  else
    begin
      let (ts,_) = Dllist.get_data inf.olast in
      Printf.printf "last (ts) = %s; " (MFOTL.string_of_ts ts)
    end;
  print_string "oauxrels = ";
  Misc.print_dllist print_sauxel inf.oauxrels;
  Sliding.print_stree MFOTL.string_of_ts (Relation.print_rel "") ";\n oinf.tree = " inf.otree;
  print_string "}"


let print_sainf str inf =
  print_string str;
  print_ainf "{srel2 = " inf.sarel2;
  Relation.print_rel "; sres=" inf.sres;
  print_string "; sauxrels=";
  Misc.print_mqueue print_sauxel inf.saauxrels;
  print_string "}"

let print_sinf str inf =
  print_string str;
  print_ainf "{srel2=" inf.srel2  ;
  print_string ", sauxrels=";
  Misc.print_mqueue print_sauxel inf.sauxrels;
  print_string "}"


let print_uinf str inf =
  Printf.printf "%s{first=%b; " str inf.ufirst;
  if inf.ulast == NEval.void then
    print_string "last=None; "
  else
    begin
      let (i,tsi) = NEval.get_data inf.ulast in
      Printf.printf "last=(%d,%s); " i (MFOTL.string_of_ts tsi)
    end;
  Relation.print_rel "res=" inf.ures;
  print_string "; raux=";
  Misc.print_dllist print_rauxel inf.raux;
  print_string "; saux=";
  Misc.print_dllist print_auxel inf.saux;
  print_endline "}"

let print_uninf str uninf =
  let get_last last =
    if last == NEval.void then "None"
    else
      begin
        let i,tsi = NEval.get_data last in
        Printf.sprintf "(%d,%s)" i (MFOTL.string_of_ts tsi)
      end
  in
  Printf.printf "%s{last1=%s; last2=%s; " str
    (get_last uninf.last1) (get_last uninf.last2);
  print_string "listrel1=";
  Misc.print_dllist print_aauxel uninf.listrel1;
  print_string "; listrel2=";
  Misc.print_dllist print_aauxel uninf.listrel2;
  print_string "}\n"

let print_ezinf str inf =
  Printf.printf "%s{" str;
  if inf.ezlastev == NEval.void then
    print_string "ezlastev = None; "
  else
    begin
      let (i,tsi) = NEval.get_data inf.ezlastev in
      Printf.printf "ezlastev = (%d,%s); " i (MFOTL.string_of_ts tsi)
    end;
  if inf.ezlast == Dllist.void then
    print_string "ezlast = None; "
  else
    begin
      let (_,ts,_) = Dllist.get_data inf.ezlast in
      Printf.printf "elast (ts) = %s; " (MFOTL.string_of_ts ts)
    end;
  print_string "eauxrels=";
  Misc.print_dllist print_aauxel inf.ezauxrels;
  Sliding.print_stree string_of_int (Relation.print_rel "") "; ezinf.eztree = " inf.eztree;
  print_string "}\n"


let print_einf str inf =
  Printf.printf "%s{" str;
  if inf.elastev == NEval.void then
    print_string "elastev = None; "
  else
    begin
      let (i,tsi) = NEval.get_data inf.elastev in
      Printf.printf "elastev = (%d,%s); " i (MFOTL.string_of_ts tsi)
    end;
  if inf.elast == Dllist.void then
    print_string "elast = None; "
  else
    begin
      let ts = fst (Dllist.get_data inf.elast) in
      Printf.printf "elast (ts) = %s; " (MFOTL.string_of_ts ts)
    end;
  print_string "eauxrels=";
  Misc.print_dllist print_sauxel inf.eauxrels;
  Sliding.print_stree MFOTL.string_of_ts (Relation.print_rel "") "; einf.etree = " inf.etree;
  print_string "}"

let print_einfn str inf =
  print_einf str inf;
  print_newline()


let print_extf str ff =
  let print_spaces d =
    for i = 1 to d do print_string " " done
  in
  let rec print_f_rec d f =
    print_spaces d;
    (match f with
     | ERel _ ->
       print_string "ERel\n";

     | EPred (p,_,inf) ->
       Predicate.print_predicate p;
       print_string ": inf=";
       print_inf inf;
       print_string "\n"

     | _ ->
       (match f with
        | ENeg f ->
          print_string "NOT\n";
          print_f_rec (d+1) f;

        | EExists (_,f) ->
          print_string "EXISTS\n";
          print_f_rec (d+1) f;

        | EPrev (intv,f,pinf) ->
          print_string "PREVIOUS";
          MFOTL.print_interval intv;
          print_string ": ptsq=";
          MFOTL.print_ts pinf.ptsq;
          print_string "\n";
          print_f_rec (d+1) f

        | ENext (intv,f,ninf) ->
          print_string "NEXT";
          MFOTL.print_interval intv;
          print_string ": init=";
          print_bool ninf.init;
          print_string "\n";
          print_f_rec (d+1) f

        | EOnceA (intv,f,inf) ->
          print_string "ONCE";
          MFOTL.print_interval intv;
          Relation.print_rel ": rel = " inf.ores;
          print_string "; oaauxrels = ";
          Misc.print_mqueue print_sauxel inf.oaauxrels;
          print_string "\n";
          print_f_rec (d+1) f

        | EOnceZ (intv,f,oinf) ->
          print_string "ONCE";
          MFOTL.print_interval intv;
          print_ozinf ": ozinf=" oinf;
          print_f_rec (d+1) f

        | EOnce (intv,f,oinf) ->
          print_string "ONCE";
          MFOTL.print_interval intv;
          print_oinf ": oinf = " oinf;
          print_string "\n";
          print_f_rec (d+1) f

        | EEventuallyZ (intv,f,einf) ->
          print_string "EVENTUALLY";
          MFOTL.print_interval intv;
          print_ezinf ": ezinf=" einf;
          print_f_rec (d+1) f

        | EEventually (intv,f,einf) ->
          print_string "EVENTUALLY";
          MFOTL.print_interval intv;
          print_einf ": einf=" einf;
          print_string "\n";
          print_f_rec (d+1) f

        | _ ->
          (match f with
           | EAnd (_,f1,f2,ainf) ->
             print_ainf "AND: ainf=" ainf.arel;
             print_string "\n";
             print_f_rec (d+1) f1;
             print_f_rec (d+1) f2

           | EOr (_,f1,f2,ainf) ->
             print_ainf "OR: ainf=" ainf.arel;
             print_string "\n";
             print_f_rec (d+1) f1;
             print_f_rec (d+1) f2

           | ESinceA (_,intv,f1,f2,sinf) ->
             print_string "SINCE";
             MFOTL.print_interval intv;
             print_sainf ": sinf = " sinf;
             print_string "\n";
             print_f_rec (d+1) f1;
             print_f_rec (d+1) f2

           | ESince (_,intv,f1,f2,sinf) ->
             print_string "SINCE";
             MFOTL.print_interval intv;
             print_sinf ": sinf=" sinf;
             print_string "\n";
             print_f_rec (d+1) f1;
             print_f_rec (d+1) f2

           | EUntil (_,intv,f1,f2,uinf) ->
             print_string "UNTIL";
             MFOTL.print_interval intv;
             print_uinf ": uinf=" uinf;
             print_f_rec (d+1) f1;
             print_f_rec (d+1) f2

           | ENUntil (_,intv,f1,f2,uninf) ->
             print_string "NUNTIL";
             MFOTL.print_interval intv;
             print_uninf ": uninf=" uninf;
             print_f_rec (d+1) f1;
             print_f_rec (d+1) f2

           | _ -> failwith "[print_formula] internal error"
          );
       );
    );
  in
  print_string str;
  print_f_rec 0 ff








let mqueue_add_last auxrels tsq rel2 =
  if Mqueue.is_empty auxrels then
    Mqueue.add (tsq,rel2) auxrels
  else
    let tslast, rellast =  Mqueue.get_last auxrels in
    if tslast = tsq then
      Mqueue.update_last (tsq, Relation.union rellast rel2) auxrels
    else
      Mqueue.add (tsq,rel2) auxrels

let dllist_add_last auxrels tsq rel2 =
  if Dllist.is_empty auxrels then
    Dllist.add_last (tsq,rel2) auxrels
  else
    let tslast, rellast = Dllist.get_last auxrels in
    if tslast = tsq then
      let _ = Dllist.pop_last auxrels in
      Dllist.add_last (tsq, Relation.union rellast rel2) auxrels
    else
      Dllist.add_last (tsq,rel2) auxrels





(* [saauxrels] consists of those relations that are outside of the
   relevant time window *)
let update_since_all intv tsq inf comp rel1 rel2 =
  inf.sres <- comp inf.sres rel1;
  let auxrels = inf.saauxrels in
  let rec elim () =
    if not (Mqueue.is_empty auxrels) then
      let (tsj,relj) = Mqueue.top auxrels in
      if MFOTL.in_right_ext (MFOTL.ts_minus tsq tsj) intv then
        begin
          ignore (Mqueue.pop auxrels);
          inf.sres <- Relation.union inf.sres (comp relj rel1);
          elim ()
        end
  in
  elim ();

  Mqueue.update_and_delete
    (fun (tsj, relj) -> (tsj, comp relj rel1))
    (fun (_,relj) -> Relation.is_empty relj) (* delete the current node if newrel is empty *)
    auxrels;

  if not (Relation.is_empty rel2) then
    begin
      if MFOTL.in_right_ext MFOTL.ts_null intv then
        inf.sres <- Relation.union inf.sres rel2;
      mqueue_add_last auxrels tsq rel2
    end;

  inf.sres



let update_since intv tsq auxrels comp discard rel1 rel2 =
  let rec elim_old_auxrels () =
    (* remove old elements that felt out of the interval *)
    if not (Mqueue.is_empty auxrels) then
      let (tsj,relj) = Mqueue.top auxrels in
      if not (MFOTL.in_left_ext (MFOTL.ts_minus tsq tsj) intv) then
        begin
          ignore(Mqueue.pop auxrels);
          elim_old_auxrels()
        end
  in
  elim_old_auxrels ();

  let res = ref Relation.empty in
  Mqueue.update_and_delete
    (fun (tsj,relj) ->
       let newrel = comp relj rel1 in
       if (not discard) && MFOTL.in_right_ext (MFOTL.ts_minus tsq tsj) intv then
         res := Relation.union !res newrel;
       (tsj,newrel)
    )
    (* delete the current node if newrel is empty *)
    (fun (_,relj) -> Relation.is_empty relj)
    auxrels;

  if not (Relation.is_empty rel2) then
    begin
      if (not discard) && MFOTL.in_right_ext MFOTL.ts_null intv then
        res := Relation.union !res rel2;
      mqueue_add_last auxrels tsq rel2
    end;

  !res


let update_once_all intv tsq inf =
  let auxrels = inf.oaauxrels in
  let rec comp () =
    if not (Mqueue.is_empty auxrels) then
      let (tsj,relj) = Mqueue.top auxrels in
      if MFOTL.in_right_ext (MFOTL.ts_minus tsq tsj) intv then
        begin
          ignore (Mqueue.pop auxrels);
          inf.ores <- Relation.union inf.ores relj;
          comp ()
        end
  in
  comp ()




(* It returns the list consisting of the new elements in the new time
   window with respect to the old time window. It is used by once and
   eventually evaluation functions.

   Arguments:
   - [l] the (doubly-linked) list of old elements
   - [last] a pointer to the element of the list from which the
   processing starts
   - [cond] stopping condition
   - [f] a function to be applied on each element
*)
let get_new_elements l last cond f =
  let rec get crt new_last acc =
    let v = Dllist.get_data crt in
    if cond v then
      if Dllist.is_last l crt then
        (f v) :: acc, crt
      else
        get (Dllist.get_next l crt) crt ((f v) :: acc)
    else
      acc, new_last
  in
  if last == Dllist.void then
    get (Dllist.get_first_cell l) Dllist.void []
  else if not (Dllist.is_last l last) then
    get (Dllist.get_next l last) last []
  else
    [], last


(* Remark: we could remove all auxrels that are covered by the tree and
   gain some memory (sooner). However detecting [lw] would be harder. *)
let update_once_zero intv q tsq inf rel2 discard =
  let auxrels = inf.ozauxrels in

  let rec elim_old_ozauxrels () =
    (* remove old elements that fell out of the interval *)
    if not (Dllist.is_empty auxrels) then
      let (_, tsj, arel) = Dllist.get_first auxrels in
      if not (MFOTL.in_left_ext (MFOTL.ts_minus tsq tsj) intv) then
        begin
          if inf.ozlast != Dllist.void && inf.ozlast == Dllist.get_first_cell auxrels then
            inf.ozlast <- Dllist.void;
          ignore(Dllist.pop_first auxrels);
          elim_old_ozauxrels()
        end
  in
  elim_old_ozauxrels ();

  if not (Relation.is_empty rel2) then
    Dllist.add_last (q,tsq,rel2) inf.ozauxrels;

  if Dllist.is_empty auxrels || discard then
    Relation.empty
  else
    let cond = fun _ -> true in
    let f = fun (j,_,rel) -> (j,rel) in
    let subseq, new_last = get_new_elements auxrels inf.ozlast cond f in
    let lw,_,_ = Dllist.get_first auxrels in
    let rw =
      if subseq = [] then
        let j,_,_ = Dllist.get_data inf.ozlast in j
      else
        begin
          assert (new_last != Dllist.void);
          inf.ozlast <- new_last;
          let rw = fst (List.hd subseq) in
          assert (rw = let j,_,_ = Dllist.get_data new_last in j);
          rw
        end
    in
    if Misc.debugging Dbg_eval then
      begin
        Printf.printf "[update_once_zero] lw = %d rw = %d " lw rw;
        Misc.printnl_list "subseq = " print_auxel subseq;
      end;
    let newt = Sliding.slide string_of_int Relation.union subseq (lw, rw) inf.oztree in
    inf.oztree <- newt;
    Sliding.stree_res newt


let update_once intv tsq inf discard =
  let auxrels = inf.oauxrels in
  let rec elim_old_oauxrels () =
    (* remove old elements that fell out of the interval *)
    if not (Dllist.is_empty auxrels) then
      let (tsj,_) = Dllist.get_first auxrels in
      if not (MFOTL.in_left_ext (MFOTL.ts_minus tsq tsj) intv) then
        begin
          if inf.olast != Dllist.void && inf.olast == Dllist.get_first_cell auxrels then
            inf.olast <- Dllist.void;
          ignore(Dllist.pop_first auxrels);
          elim_old_oauxrels()
        end
  in
  elim_old_oauxrels ();

  (* In the following we distiguish between the new window and the new
     elements: the new window may contain old elements (the old and new
     windows may overlap). *)

  if Dllist.is_empty auxrels || discard then
    Relation.empty
  else
    let lw = fst (Dllist.get_first auxrels) in
    if MFOTL.in_right_ext (MFOTL.ts_minus tsq lw) intv then
      (* the new window is not empty *)
      let cond = fun (tsj,_) -> MFOTL.in_right_ext (MFOTL.ts_minus tsq tsj) intv in
      let subseq, new_last = get_new_elements auxrels inf.olast cond (fun x -> x) in
      let rw =
        if subseq = [] then
          fst (Dllist.get_data inf.olast)
        else
          begin
            assert (new_last != Dllist.void);
            inf.olast <- new_last;
            let rw = fst (List.hd subseq) in
            assert (rw = fst (Dllist.get_data new_last));
            rw
          end
      in
      if Misc.debugging Dbg_eval then
        begin
          Printf.printf "[update_once] lw = %s rw = %s "
            (MFOTL.string_of_ts lw)
            (MFOTL.string_of_ts rw);
          Misc.printnl_list "subseq = " print_sauxel subseq;
        end;
      let newt = Sliding.slide MFOTL.string_of_ts Relation.union subseq (lw, rw) inf.otree in
      inf.otree <- newt;
      Sliding.stree_res newt
    else
      begin
        (* the new window is empty,
           because not even the oldest element satisfies the constraint *)
        inf.otree <- LNode {l = MFOTL.ts_invalid;
                            r = MFOTL.ts_invalid;
                            res = Some (Relation.empty)};
        inf.olast <- Dllist.void;
        Relation.empty
      end





let update_old_until q tsq i intv inf discard  =
  (* eliminate those entries (q-1,reli) from rels;
     return the tuples which hold at q *)
  let elim_old j rels =
    assert(j>=q-1);
    if not (Sk.is_empty rels) then
      let (k,relk) = Sk.get_first rels in
      if k=q-1 then
        begin
          ignore(Sk.pop_first rels);
          if not (Sk.is_empty rels) then
            let (k',relk') = Sk.pop_first rels in
            assert(k'>=q && j>=q);
            let newrelk' = Relation.union relk relk' in
            Sk.add_first (k',newrelk') rels;
            if k'=q then
              newrelk'
            else
              relk
          else
          if (j>q-1) then
            begin
              Sk.add_first (k+1,relk) rels;
              relk
            end
          else
            Relation.empty
        end
      else
        begin
          assert(k>q-1);
          if k=q then
            relk
          else
            Relation.empty
        end
    else (* Sk.is_empty rels = true *)
      Relation.empty
  in


  let rec elim_old_raux () =
    (* remove old elements that fell out of the interval *)
    if not (Sj.is_empty inf.raux) then
      let (j,tsj,_) = Sj.get_first inf.raux in
      if j<q || not (MFOTL.in_right_ext (MFOTL.ts_minus tsj tsq) intv) then
        begin
          ignore(Sj.pop_first inf.raux);
          elim_old_raux()
        end
  in

  elim_old_raux ();

  Sj.iter (
    fun (j,tsj,rrels) ->
      assert(j>=q);
      assert(MFOTL.in_right_ext (MFOTL.ts_minus tsj tsq) intv);
      let relq = elim_old j rrels in
      if (not discard) && not (Relation.is_empty relq) then
        inf.ures <- Relation.union inf.ures relq;
      if Misc.debugging Dbg_eval then
        Relation.print_reln "[update_aux] res: " inf.ures;
  ) inf.raux;

  (* saux holds elements (k,relk) for the last seen index,
     i.e. [i] *)
  assert(i>=q-1);
  if i=q-1 then
    Sk.clear inf.saux
  else
    ignore(elim_old i inf.saux)


(* Auxiliary functions for the f1 Until_I f2 case.

   The saux list contains tuples (k,Sk) (ordered incrementally by k),
   with q <= k <= i, such that the tuples in Sk satisfy f1
   continuously between k and i, and k is minimal (that is, if a tuple
   is in Sk it will not also be in Sk' with k'>k.)

   The raux list contains tuples (j,tj,Lj) (ordered incrementaly by
   j), with q <= j <= i, where Lj is a list of tuples (k,Rk) (ordered
   incrementaly by k), with q <= k <= j, such that the tuples in Rk
   satisfy f2 at j and satisfy f1 continuously between k and j-1, and
   k is minimal (that is, if a tuple is in Rk it will not also be in
   Rk' with j>=k'>k.)

   NOTE: The iteration through raux to eliminate those tuples <k,Sk>
   with k<q (ie. k=q-1) seems unnecessary. If a tuple in Sk satisfies
   f1 continuously from k to j, then it also satisfies f1 continuously
   from q to j.
*)


let combine2 comp j rels rel2 =
  let nrels = Sk.empty() in
  let curr_rel2 = ref rel2 in
  Sk.iter
    (fun (k,rel) ->
       let nrel = comp !curr_rel2 rel in
       if not (Relation.is_empty nrel) then
         Sk.add_last (k,nrel) nrels;
       curr_rel2 := Relation.diff !curr_rel2 nrel;
    ) rels;
  if not (Relation.is_empty !curr_rel2) then
    Sk.add_last (j,!curr_rel2) nrels;
  nrels

let get_relq q rels =
  if not (Sj.is_empty rels) then
    let (k,relk) = Sj.get_first rels in
    if k = q then Some relk
    else None
  else
    None

let update_until q tsq i tsi intv rel1 rel2 inf comp discard =
  if Misc.debugging Dbg_eval then
    print_uinf "[update_until] inf: " inf;
  assert(i >= q);
  let nsaux = combine2 Relation.inter i inf.saux rel1 in
  if (MFOTL.in_right_ext (MFOTL.ts_minus tsi tsq) intv) &&
     not (Relation.is_empty rel2) then
    begin
      let rrels = combine2 comp i inf.saux rel2 in
      Sj.add_last (i,tsi,rrels) inf.raux;
      if not discard then
        match get_relq q rrels with
        | Some rel -> inf.ures <- Relation.union inf.ures rel
        | None -> ()
    end;
  inf.saux <- nsaux


let elim_old_eventually q tsq intv inf =
  let auxrels = inf.eauxrels in

  let rec elim_old_eauxrels () =
    (* remove old elements that fell out of the interval *)
    if not (Dllist.is_empty auxrels) then
      let (tsj, _) = Dllist.get_first auxrels in
      if not (MFOTL.in_right_ext (MFOTL.ts_minus tsj tsq) intv) then
        begin
          if inf.elast != Dllist.void && inf.elast == Dllist.get_first_cell auxrels then
            inf.elast <- Dllist.void;
          ignore(Dllist.pop_first auxrels);
          elim_old_eauxrels()
        end
  in

  elim_old_eauxrels ()



(* Instead of a single Dllist and a pointer, which says where in this
   list end the current time window, we use two queues. One queue
   stores the time window, the other ones stores the relations not
   yet in the time window *)
let comp_agg_once tsq intv state update_state_old update_state_new get_result rel discard =
  let rec elim_old_from_timewindow () =
    (* remove old elements that fell out of the interval *)
    if not (Queue.is_empty state.tw_rels) then
      let (tsj, arel) = Queue.top state.tw_rels in
      if not (MFOTL.in_left_ext (MFOTL.ts_minus tsq tsj) intv) then
        begin
          ignore(Queue.pop state.tw_rels);
          update_state_old state arel;
          elim_old_from_timewindow ()
        end
  in

  let rec consider_other_rels () =
    if not (Queue.is_empty state.other_rels) then
      begin
        let (tsj, arel) = Queue.top state.other_rels in
        let diff = MFOTL.ts_minus tsq tsj in
        if not (MFOTL.in_left_ext diff intv) then
          begin
            (* current relation already too old for the new time window *)
            ignore (Queue.pop state.other_rels);
            consider_other_rels ()
          end
        else if MFOTL.in_interval diff intv then
          (* current relation in the interval, so we process it *)
          begin
            ignore (Queue.pop state.other_rels);
            let arel' = update_state_new state arel in
            Queue.push (tsj, arel') state.tw_rels;
            consider_other_rels ()
          end
          (* else, that is, not (MFOTL.in_right_ext diff intv) *)
          (* current relation too new, so we stop and consider it next time *)
      end
  in

  elim_old_from_timewindow ();
  if not (Relation.is_empty rel) then
    Queue.push (tsq, rel) state.other_rels;
  consider_other_rels ();
  if not discard then
    get_result state
  else
    Relation.empty



let comp_aggMM_once tsq intv state update_state_old update_state_new get_result rel discard =
  let rec consider_other_rels () =
    if not (Queue.is_empty state.non_tw_rels) then
      begin
        let (tsj, arel) = Queue.top state.non_tw_rels in
        let diff = MFOTL.ts_minus tsq tsj in
        if not (MFOTL.in_left_ext diff intv) then
          begin
            (* current relation already too old for the new time window *)
            ignore (Queue.pop state.non_tw_rels);
            consider_other_rels ()
          end
        else if MFOTL.in_interval diff intv then
          (* current relation in the interval, so we process it *)
          begin
            ignore (Queue.pop state.non_tw_rels);
            update_state_new state tsq arel;
            consider_other_rels ()
          end
          (* else, that is, not (MFOTL.in_right_ext diff intv) *)
          (* current relation too new, so we stop and consider it next time *)
      end
  in

  update_state_old state tsq;
  if not (Relation.is_empty rel) then
    Queue.push (tsq, rel) state.non_tw_rels;
  consider_other_rels ();
  if not discard then
    get_result state
  else
    Relation.empty



(* Is 'last' pointing to the last position in neval? *)
(* This NEval.void hack is ugly, but it seems unavoidable, unless we
   have a separate [neval] for each subformula *)
let neval_is_last neval last =
  (not (last == NEval.void)) && NEval.is_last neval last

(* get the current position to be evaluated (for some subformula) *)
let neval_get_crt neval last crt q =
  if last == NEval.void then
    begin
      assert(q=0);
      crt
    end
  else
    NEval.get_next neval last


(* Arguments:
   - [f] the current formula
   - [neval] the list of non-evaluated points
   - [crt] the current evaluation point (a time point, timestamp pair)
   - [discard] a boolean; if true then the result is not used
               (only a minimal amount of computation should be done);
               it should not be propagated for temporal subformulas
               (pitfall: possible source of bugs)
*)
let rec eval f neval crt discard =
  let (q,tsq) = NEval.get_data crt in

  if Misc.debugging Dbg_eval then
    begin
      print_extf "\n[eval] evaluating formula\n" f;
      Printf.printf "at (%d,%s) with discard=%b and " q (MFOTL.string_of_ts tsq) discard;
      print_neval "neval=" neval
    end;

  match f with
  | ERel rel -> Some rel

  | EPred (p,_,inf) ->
    if Misc.debugging Dbg_eval then
      begin
        print_string "[eval,Pred] ";
        Predicate.print_predicate p;
        print_predinf  ": " inf
      end;

    let (cq,ctsq,rel) = Queue.pop inf in
    assert (cq = q && ctsq = tsq);
    Some rel

  | ENeg f1 ->
    (match eval f1 neval crt discard with
     | Some rel ->
       let res =
         if Relation.is_empty rel then (* false? *)
           Relation.singleton (Tuple.make_tuple [])
         else
           Relation.empty (* true *)
       in
       Some res
     | None -> None
    )

  | EExists (comp,f1) ->
    (match eval f1 neval crt discard with
     | Some rel -> Some (comp rel)
     | None -> None
    )

  | EAnd (comp,f1,f2,inf) ->
    (* we have to store rel1, if f2 cannot be evaluated *)
    let eval_and rel1 =
      if Relation.is_empty rel1 then
        (match eval f2 neval crt true with
         | Some _ ->
           inf.arel <- None;
           Some rel1
         | None ->
           inf.arel <- Some rel1;
           None
        )
      else
        (match eval f2 neval crt discard with
         | Some rel2 ->
           inf.arel <- None;
           Some (comp rel1 rel2)
         | None ->
           inf.arel <- Some rel1;
           None
        )
    in
    (match inf.arel with
     | Some rel1 -> eval_and rel1
     | None ->
       (match eval f1 neval crt discard with
        | Some rel1 -> eval_and rel1
        | None -> None
       )
    )

  | EAggreg (comp, f) ->
    (match eval f neval crt discard with
     | Some rel -> Some (comp rel)
     | None -> None
    )

  | EOr (comp, f1, f2, inf) ->
    (* we have to store rel1, if f2 cannot be evaluated *)
    (match inf.arel with
     | Some rel1 ->
       (match eval f2 neval crt discard with
        | Some rel2 ->
          inf.arel <- None;
          Some (comp rel1 rel2)
        | None -> None
       )
     | None ->
       (match eval f1 neval crt discard with
        | Some rel1 ->
          (match eval f2 neval crt discard with
           | Some rel2 -> Some (comp rel1 rel2)
           | None ->
             inf.arel <- Some rel1;
             None
          )
        | None -> None
       )
    )

  | EPrev (intv,f1,inf) ->
    if Misc.debugging Dbg_eval then
      Printf.printf "[eval,Prev] inf.ptsq=%s\n%!" (MFOTL.string_of_ts inf.ptsq);

    if q=0 then
      begin
        inf.ptsq <- tsq;
        Some Relation.empty
      end
    else
      begin
        assert(not (inf.ptsq = MFOTL.ts_invalid && NEval.is_first neval crt));
        let added = ref false in
        if NEval.is_first neval crt then
          begin
            NEval.add_first (q-1,inf.ptsq) neval;
            added := true;
          end;
        let pcrt = NEval.get_prev neval crt in
        begin
          let prev_tp, prev_ts = NEval.get_data pcrt in
          assert(prev_tp = q - 1);
          assert(prev_ts = inf.ptsq);
          let orel = eval f1 neval pcrt discard in
          if !added then
            ignore(NEval.pop_first neval);
          match orel with
          | Some rel1 ->
            let res =
              if MFOTL.in_interval (MFOTL.ts_minus tsq inf.ptsq) intv then
                Some rel1
              else
                Some Relation.empty
            in
            inf.ptsq <- tsq;
            res
          | None -> None
        end
      end

  | ENext (intv,f1,inf) ->
    if Misc.debugging Dbg_eval then
      Printf.printf "[eval,Next] inf.init=%b\n%!" inf.init;

    if inf.init then
      begin
        match eval f1 neval crt discard with
        | Some _ -> inf.init <- false
        | _ -> ()
      end;

    if NEval.is_last neval crt then
      None
    else
      begin
        (* ignore(NEval.pop_first neval); *)
        let ncrt = NEval.get_next neval crt in
        let orel = eval f1 neval ncrt discard in
        (* NEval.add_first (q,tsq) neval; *)
        match orel with
        | Some rel1 ->
          let (nq,ntsq) = NEval.get_data ncrt in
          assert(nq=q+1);
          if MFOTL.in_interval (MFOTL.ts_minus ntsq tsq) intv then
            Some rel1
          else
            Some Relation.empty
        | None -> None
      end

  | ESinceA (comp,intv,f1,f2,inf) ->
    if Misc.debugging Dbg_eval then
      Printf.printf "[eval,SinceA] q=%d\n%!" q;

    let eval_f1 rel2 comp2 =
      (match eval f1 neval crt false with
       | Some rel1 ->
         inf.sarel2 <- None;
         Some (comp2 rel1 rel2)
       | None ->
         inf.sarel2 <- Some rel2;
         None
      )
    in

    let update_sauxrels = update_since_all intv tsq inf comp in

    (match inf.sarel2 with
     | Some rel2 -> eval_f1 rel2 update_sauxrels
     | None ->
       (match eval f2 neval crt false with
        | None -> None
        | Some rel2 -> eval_f1 rel2 update_sauxrels
       )
    )

  | ESince (comp,intv,f1,f2,inf) ->
    if Misc.debugging Dbg_eval then
      Printf.printf "[eval,Since] q=%d\n" q;

    let eval_f1 rel2 comp2 =
      (match eval f1 neval crt false with
       | Some rel1 ->
         inf.srel2 <- None;
         Some (comp2 rel1 rel2)
       | None ->
         inf.srel2 <- Some rel2;
         None
      )
    in

    let update_sauxrels = update_since intv tsq inf.sauxrels comp discard in

    (match inf.srel2 with
     | Some rel2 -> eval_f1 rel2 update_sauxrels
     | None ->
       (match eval f2 neval crt false with
        | None -> None
        | Some rel2 -> eval_f1 rel2 update_sauxrels
       )
    )


  | EOnceA ((c,_) as intv, f2, inf) ->
    (match eval f2 neval crt false with
     | None -> None
     | Some rel2 ->
       if Misc.debugging Dbg_eval then
         Printf.printf "[eval,OnceA] q=%d\n" q;

       if c = CBnd MFOTL.ts_null then
         begin
           inf.ores <- Relation.union inf.ores rel2;
           Some inf.ores
         end
       else
         begin
           if not (Relation.is_empty rel2) then
             mqueue_add_last inf.oaauxrels tsq rel2;

           update_once_all intv tsq inf;
           Some inf.ores
         end
    )

  | EAggOnce (f, intv, state, update_old, update_new, get_result) ->
    (match eval f neval crt false with
     | Some rel -> Some (comp_agg_once
                           tsq intv state
                           update_old
                           update_new
                           get_result
                           rel discard)
     | None -> None
    )

  | EAggMMOnce (f, intv, state, update_old, update_new, get_result) ->
    (match eval f neval crt false with
     | Some rel -> Some (comp_aggMM_once
                           tsq intv state
                           update_old
                           update_new
                           get_result
                           rel discard)
     | None -> None
    )





  (* We distinguish between whether the left margin of [intv] is
     zero or not, as we need to have two different ways of
     representing the margins of the windows in the tree: when 0
     is not included we can use the timestamps and merge
     relations at equal timestamps; otherwise, when 0 is not
     included, we need to use the timepoints. *)
  | EOnceZ (intv,f2,inf) ->
    (match eval f2 neval crt false with
     | None -> None
     | Some rel2 ->
       if Misc.debugging Dbg_eval then
         Printf.printf "[eval,OnceZ] q=%d\n" q;

       Some (update_once_zero intv q tsq inf rel2 discard)
    )

  | EOnce (intv,f2,inf) ->
    (match eval f2 neval crt false with
     | None -> None
     | Some rel2 ->
       if Misc.debugging Dbg_eval then
         Printf.printf "[eval,Once] q=%d\n" q;

       if not (Relation.is_empty rel2) then
         dllist_add_last inf.oauxrels tsq rel2;

       Some (update_once intv tsq inf discard)
    )

  | EUntil (comp,intv,f1,f2,inf) ->
    (* contents of inf:  (f = f1 UNTIL_intv f2)
       ulast:        last cell of neval for which both f1 and f2 are evaluated
       ufirst:       boolean flag indicating if we are at the first
                     iteration after the evaluation of f (i.e. q was
                     just moved); in this case we remove auxiliary
                     relations at old q
       ures:         the current partial result (for f)
       urel2:        the evaluation of f2 at ulast
       raux, saux:   the auxiliary relations
    *)

    if Misc.debugging Dbg_eval then
      begin
        let str = Printf.sprintf "[eval,Until] q=%d inf: " q in
        print_uinf str inf
      end;

    if inf.ufirst then
      begin
        inf.ufirst <- false;
        assert(inf.ulast != NEval.void);
        let (i,_) = NEval.get_data inf.ulast in
        update_old_until q tsq i intv inf discard;
        if Misc.debugging Dbg_eval then
          print_uinf "[eval,Until,after_update] inf: " inf
      end;

    (* we first evaluate f2, and then f1 *)

    let rec evalf1 i tsi rel2 ncrt =
      (match eval f1 neval ncrt false with
       | Some rel1 ->
         update_until q tsq i tsi intv rel1 rel2 inf comp discard;
         inf.urel2 <- None;
         inf.ulast <- ncrt;
         evalf2 ()
       | None ->
         inf.urel2 <- (Some rel2);
         None
      )

    and evalf2 () =
      if neval_is_last neval inf.ulast then
        None
      else
        let ncrt = neval_get_crt neval inf.ulast crt q in
        let (i,tsi) = NEval.get_data ncrt in
        if not (MFOTL.in_left_ext (MFOTL.ts_minus tsi tsq) intv) then
          (* we have the lookahead, we can compute the result *)
          begin
            if Misc.debugging Dbg_eval then
              Printf.printf "[eval,Until] evaluation possible q=%d tsq=%s\n"
                q (MFOTL.string_of_ts tsq);
            let res = inf.ures in
            inf.ures <- Relation.empty;
            inf.ufirst <- true;
            Some res
          end
        else
          begin
            (match inf.urel2 with
             | Some rel2 -> evalf1 i tsi rel2 ncrt
             | None ->
               (match eval f2 neval ncrt false with
                | None -> None
                | Some rel2 -> evalf1 i tsi rel2 ncrt
               )
            )
          end
    in
    evalf2()

  | ENUntil (comp,intv,f1,f2,inf) ->
    (* contents of inf:  (f = NOT f1 UNTIL_intv f2)
       ulast1:       last cell of neval for which f1 is evaluated
       ulast2:       last cell of neval for which f2 is evaluated
       listrel1:     list of evaluated relations for f1
       listrel2:     list of evaluated relations for f2

       NOTE: a possible optimization would be to not store empty relations
    *)

    (* evaluates the subformula f as much as possible *)
    let rec eval_subf f list last  =
      if neval_is_last neval last then
        last
      else
        let ncrt = neval_get_crt neval last crt q in
        match eval f neval ncrt false with
        | None -> last
        | Some rel ->
          (* store the result and try the next time point *)
          let i, tsi = NEval.get_data ncrt in
          Dllist.add_last (i, tsi, rel) list;
          eval_subf f list ncrt
    in

    (* evaluate the two subformulas *)
    inf.last1 <- eval_subf f1 inf.listrel1 inf.last1;
    inf.last2 <- eval_subf f2 inf.listrel2 inf.last2;

    if inf.last1 == NEval.void || inf.last2 == NEval.void then
      None
    else
      let (i1,tsi1) = NEval.get_data inf.last1 in
      let (i2,tsi2) = NEval.get_data inf.last2 in
      if not (MFOTL.in_left_ext (MFOTL.ts_minus tsi2 tsq) intv) && i1 >= i2-2 then
        (* we have the lookahead, we can compute the result; note that
           that i2-1 is the last time point in the relevant interval,
           and thus the last time point for which we need f1's
           evaluation is i2-2 *)
        begin
          (* we iteratively compute the union of the relations [f1]_j
             with q <= j <= j0-1, where j0 is the first index which
             satisfies the temporal constraint relative to q *)
          let f1union = ref Relation.empty in
          let crt1_j = ref (Dllist.get_first_cell inf.listrel1) in
          let rec iter1 () =
            let j,tsj,relj = Dllist.get_data !crt1_j in
            if j < q then
              begin (* clean up from previous evaluation *)
                assert (j = q-1);
                ignore(Dllist.pop_first inf.listrel1);
                if not (Dllist.is_empty inf.listrel1) && not (Dllist.is_last inf.listrel1 !crt1_j) then
                  begin
                    crt1_j := Dllist.get_next inf.listrel1 !crt1_j;
                    iter1 ()
                  end
              end
            else if not (MFOTL.in_right_ext (MFOTL.ts_minus tsj tsq) intv) then
              begin
                f1union := Relation.union !f1union relj;
                if not (Dllist.is_last inf.listrel1 !crt1_j) then
                  begin
                    crt1_j := Dllist.get_next inf.listrel1 !crt1_j;
                    iter1 ()
                  end
              end
          in
          iter1 ();

          (* we now iterate through the remaining indexes, updating the
             union, and also computing the result *)
          let res = ref Relation.empty in
          let crt2_j = ref (Dllist.get_first_cell inf.listrel2) in
          let rec iter2 () =
            let j2,tsj2,rel2 = Dllist.get_data !crt2_j in
            if j2 < q || not (MFOTL.in_right_ext (MFOTL.ts_minus tsj2 tsq) intv) then
              begin (* clean up from previous evaluation *)
                ignore(Dllist.pop_first inf.listrel2);
                if not (Dllist.is_last inf.listrel2 !crt2_j) then
                  begin
                    crt2_j := Dllist.get_next inf.listrel2 !crt2_j;
                    iter2 ()
                  end
              end
            else
              begin
                let j1,tsj1,rel1 = Dllist.get_data !crt1_j in
                assert(j1 = j2);
                if MFOTL.in_left_ext (MFOTL.ts_minus tsj2 tsq) intv then
                  begin
                    let resj = comp rel2 !f1union in
                    res := Relation.union !res resj;
                    f1union := Relation.union !f1union rel1;
                    let is_last1 = Dllist.is_empty inf.listrel1 ||
                                   Dllist.is_last inf.listrel1 !crt1_j in
                    let is_last2 = Dllist.is_last inf.listrel2 !crt2_j in
                    assert (not (is_last1 && is_last2));
                    if (not is_last1) && (not is_last2) then
                      begin
                        crt1_j := Dllist.get_next inf.listrel1 !crt1_j;
                        crt2_j := Dllist.get_next inf.listrel2 !crt2_j;
                        iter2 ()
                      end
                  end
              end
          in
          iter2();
          Some !res
        end
      else
        None






  | EEventuallyZ (intv,f2,inf) ->
    (* contents of inf:
       elastev: 'a NEval.cell  last cell of neval for which f2 is evaluated
       eauxrels: info          the auxiliary relations (up to elastev)
    *)
    if Misc.debugging Dbg_eval then
      print_ezinf "[eval,EventuallyZ] inf: " inf;

    let rec ez_update () =
      if neval_is_last neval inf.ezlastev then
        None
      else
        let ncrt = neval_get_crt neval inf.ezlastev crt q in
        let (i,tsi) = NEval.get_data ncrt in
        (* Printf.printf "[eval,Eventually] e_update: ncrt.i = %d\n%!" i; *)
        if not (MFOTL.in_left_ext (MFOTL.ts_minus tsi tsq) intv) then
          (* we have the lookahead, we can compute the result *)
          begin
            if Misc.debugging Dbg_eval then
              Printf.printf "[eval,EventuallyZ] evaluation possible q=%d tsq=%s tsi=%s\n%!"
                q (MFOTL.string_of_ts tsq) (MFOTL.string_of_ts tsi);

            let auxrels = inf.ezauxrels in
            if Dllist.is_empty auxrels then
              Some Relation.empty
            else if discard then
              begin
                let lw, _, _ = Dllist.get_first auxrels in
                if lw = q then (* at next iteration this first element will be too old *)
                  begin
                    if inf.ezlast != Dllist.void && inf.ezlast == Dllist.get_first_cell auxrels then
                      inf.ezlast <- Dllist.void;
                    ignore(Dllist.pop_first auxrels);
                  end;
                Some Relation.empty
              end
            else
              begin
                if inf.ezlast != Dllist.void && inf.ezlast == Dllist.get_first_cell auxrels then
                  (* TODO: when can such a case occur? *)
                  inf.ezlast <- Dllist.void;

                let lw, _, _ = Dllist.get_first auxrels in
                assert (lw >= q); (* TODO: when lw > q *)
                let cond = fun (_,tsj,_) -> MFOTL.in_left_ext (MFOTL.ts_minus tsj tsq) intv in
                let f = fun (j,_,rel) -> (j,rel) in
                let subseq, new_last = get_new_elements auxrels inf.ezlast cond f in
                let rw =
                  if subseq = [] then
                    (* TODO: why j? when does this case occur? *)
                    let j, _, _  = Dllist.get_data inf.ezlast in j
                  else
                    begin
                      assert (new_last != Dllist.void);
                      inf.ezlast <- new_last;
                      let rw = fst (List.hd subseq) in
                      assert (rw = let j, _, _ = Dllist.get_data new_last in j);
                      rw
                    end
                in

                if Misc.debugging Dbg_eval then
                  begin
                    Printf.printf "[eval,EventuallyZ] lw = %d rw = %d " lw rw;
                    Misc.printnl_list "subseq = " print_auxel subseq;
                  end;

                let newt = Sliding.slide string_of_int Relation.union subseq (lw, rw) inf.eztree in

                if lw = q then (* at next iteration this first element will be too old *)
                  begin
                    if new_last == Dllist.get_first_cell auxrels then
                      inf.ezlast <- Dllist.void;
                    ignore(Dllist.pop_first auxrels);
                  end;

                inf.eztree <- newt;
                Some (Sliding.stree_res newt)
              end
          end
        else (* we don't have the lookahead -> we cannot compute the result *)
          begin
            match eval f2 neval ncrt false with
            | None -> None
            | Some rel2 ->
              (* we update the auxiliary relations *)
              if not (Relation.is_empty rel2) then
                Dllist.add_last (i,tsi,rel2) inf.ezauxrels;
              inf.ezlastev <- ncrt;
              ez_update ()
          end
    in
    ez_update ()


  | EEventually (intv,f2,inf) ->
    (* contents of inf:
       elastev: 'a NEval.cell  last cell of neval for which f2 is evaluated
       eauxrels: info          the auxiliary relations (up to elastev)
    *)
    if Misc.debugging Dbg_eval then
      print_einfn "[eval,Eventually] inf: " inf;

    (* we could in principle do this update less often: that is, we
       can do after each evaluation, but we need to find out the
       value of ts_{q+1} *)
    elim_old_eventually q tsq intv inf;

    let rec e_update () =
      if neval_is_last neval inf.elastev then
        None
      else
        let ncrt = neval_get_crt neval inf.elastev crt q in
        let (i,tsi) = NEval.get_data ncrt in
        (* Printf.printf "[eval,Eventually] e_update: ncrt.i = %d\n%!" i; *)
        if not (MFOTL.in_left_ext (MFOTL.ts_minus tsi tsq) intv) then
          (* we have the lookahead, we can compute the result *)
          begin
            if Misc.debugging Dbg_eval then
              Printf.printf "[eval,Eventually] evaluation possible q=%d tsq=%s tsi=%s\n%!"
                q (MFOTL.string_of_ts tsq) (MFOTL.string_of_ts tsi);

            let auxrels = inf.eauxrels in
            if Dllist.is_empty auxrels || discard then
              Some Relation.empty
            else
              let lw, _ = Dllist.get_first auxrels in
              if MFOTL.in_left_ext (MFOTL.ts_minus lw tsq) intv then
                (* the new window is not empty *)
                let cond = fun (tsj,_) -> MFOTL.in_left_ext (MFOTL.ts_minus tsj tsq) intv in
                let subseq, new_last = get_new_elements auxrels inf.elast cond (fun x -> x) in
                let rw =
                  if subseq = [] then
                    fst (Dllist.get_data inf.elast)
                  else
                    begin
                      assert (new_last != Dllist.void);
                      inf.elast <- new_last;
                      let rw = fst (List.hd subseq) in
                      assert (rw = fst (Dllist.get_data new_last));
                      rw
                    end
                in
                if Misc.debugging Dbg_eval then
                  begin
                    Printf.printf "[eval,Eventually] lw = %s rw = %s "
                      (MFOTL.string_of_ts lw)
                      (MFOTL.string_of_ts rw);
                    Misc.printnl_list "subseq = " print_sauxel subseq;
                  end;
                let newt = Sliding.slide MFOTL.string_of_ts Relation.union subseq (lw, rw) inf.etree in
                inf.etree <- newt;
                Some (Sliding.stree_res newt)
              else
                begin
                  (* the new window is empty,
                     because not even the oldest element satisfies the constraint *)
                  inf.etree <- LNode {l = MFOTL.ts_invalid;
                                      r = MFOTL.ts_invalid;
                                      res = Some (Relation.empty)};
                  inf.elast <- Dllist.void;
                  Some Relation.empty
                end
          end
        else
          begin
            match eval f2 neval ncrt false with
            | None -> None
            | Some rel2 ->
              (* we update the auxiliary relations *)
              if (MFOTL.in_right_ext (MFOTL.ts_minus tsi tsq) intv) &&
                 not (Relation.is_empty rel2) then
                dllist_add_last inf.eauxrels tsi rel2;
              inf.elastev <- ncrt;
              e_update ()
          end
    in
    e_update ()







let add_index f i tsi db =
  let rec update = function
    | EPred (p, comp, inf) ->
      let rel =
        (try
           let t = Db.get_table db p in
           Table.get_relation t
         with Not_found ->
         match Predicate.get_name p with
         | "tp" -> Relation.singleton (Tuple.make_tuple [Int i])
         | "ts" -> Relation.singleton (Tuple.make_tuple [Float tsi])
         | "tpts" ->
           Relation.singleton (Tuple.make_tuple [Int i; Float tsi])
         | _ -> Relation.empty
        )
      in
      let rel = comp rel in
      Queue.add (i,tsi,rel) inf

    | ERel _ -> ()

    | ENeg f1
    | EExists (_,f1)
    | EAggOnce (f1,_,_,_,_,_)
    | EAggMMOnce (f1,_,_,_,_,_)
    | EAggreg (_,f1)
    | ENext (_,f1,_)
    | EPrev (_,f1,_)
    | EOnceA (_,f1,_)
    | EOnceZ (_,f1,_)
    | EOnce (_,f1,_)
    | EEventuallyZ (_,f1,_)
    | EEventually (_,f1,_) ->
      update f1

    | EAnd (_,f1,f2,_)
    | EOr (_,f1,f2,_)
    | ESinceA (_,_,f1,f2,_)
    | ESince (_,_,f1,f2,_)
    | ENUntil (_,_,f1,f2,_)
    | EUntil (_,_,f1,f2,_) ->
      update f1;
      update f2
  in
  update f






(** This function displays the "results" (if any) obtained after
    analyzing event index [i]. The results are those tuples satisfying
    the formula for some index [q<=i]. *)
let rec show_results closed i q tsq rel =
  if !Misc.stop_at_first_viol && Relation.cardinal rel > 1 then
    let rel2 = Relation.singleton (Relation.choose rel) in
    show_results closed i q tsq rel2
  else if !Misc.verbose then
    if closed then
      Printf.printf "@%s (time point %d): %b\n%!"
        (MFOTL.string_of_ts tsq) q (rel <> Relation.empty)
    else
      begin
        Printf.printf "@%s (time point %d): " (MFOTL.string_of_ts tsq) q;
        Relation.print_reln "" rel
      end
  else
    begin
      if Misc.debugging Dbg_perf then
        Perf.show_results q tsq;
      if rel <> Relation.empty then (* formula satisfied *)
        if closed then (* no free variables *)
          Printf.printf "@%s (time point %d): true\n%!" (MFOTL.string_of_ts tsq) q
        else (* free variables *)
          begin
            Printf.printf "@%s (time point %d): " (MFOTL.string_of_ts tsq) q;
            Relation.print_rel4 "" rel;
            print_newline()
          end
    end



let process_index ff closed neval i =
  if !Misc.verbose then
    Printf.printf "At time point %d:\n%!" i;

  let rec eval_loop () =
    if not (NEval.is_empty neval) then
      let first = NEval.get_first_cell neval in
      let (q,tsq) = NEval.get_data first in
      if tsq < MFOTL.ts_max then
        match eval ff neval first false with
        | Some rel ->
          ignore(NEval.pop_first neval);
          show_results closed i q tsq rel;
          if !Misc.stop_at_first_viol && not (Relation.is_empty rel) then false
          else eval_loop ()
        | None -> true
      else false
    else true
  in
  eval_loop ()





let comp_aggreg init_value update posx posG rel =
  let map = ref Tuple_map.empty in
  Relation.iter
    (fun tuple ->
       let gtuple = Tuple.projections posG tuple in
       let crt_value = Tuple.get_at_pos tuple posx in
       (*   match Tuple.get_at_pos tuple posx with *)
       (*     | Int v -> v *)
       (*     | _ -> failwith "[comp_aggreg] internal error" *)
       (* in *)
       try
         let old_agg_value = Tuple_map.find gtuple !map in
         let new_agg_value = update old_agg_value crt_value in
         map := Tuple_map.add gtuple new_agg_value !map
       with
       | Not_found ->
         map := Tuple_map.add gtuple (init_value crt_value) !map;
    )
    rel;
  !map

let comp_aggreg init_value update posx posG rel =
  let map = Hashtbl.create 1000 in
  Relation.iter
    (fun tuple ->
       let gtuple = Tuple.projections posG tuple in
       let crt_value = Tuple.get_at_pos tuple posx in
       (*   match Tuple.get_at_pos tuple posx with *)
       (*     | Int v -> v *)
       (*     | _ -> failwith "[comp_aggreg] internal error" *)
       (* in *)
       try
         let old_agg_value = Hashtbl.find map gtuple in
         let new_agg_value = update old_agg_value crt_value in
         Hashtbl.replace map gtuple new_agg_value;
       with
       | Not_found ->
         Hashtbl.add map gtuple (init_value crt_value);
    )
    rel;
  map

exception Break


let size xlist =
  let s = ref 0 in
  IntMap.iter (fun _ m ->
      assert(m > 0);
      s := !s + m;
    ) xlist;
  !s


(* The following assumptionn should hold: [len] is the sum of all
   bindings [m] in [xlist] *)
let median xlist len fmed =
  assert (len <> 0);
  assert (len = size xlist);
  let mid = if len mod 2 = 0 then (len / 2) - 1 else len / 2 in
  let flag = ref false in
  let crt = ref 0 in
  let med = ref (fst (IntMap.choose xlist)) in
  let prev = ref !med in
  try
    IntMap.iter (fun c m ->
        if !flag then
          begin med := fmed !prev c; raise Break end
        else
        if mid < !crt + m then (* c is the (left) median *)
          if len mod 2 = 0 then
            if mid = !crt + m - 1 then
              begin flag := true;  prev := c end
            else
              begin med := c; (* that is, (c+c)/2 *) raise Break end
          else begin med := c; raise Break end
        else
          crt := !crt + m
      ) xlist;
    failwith "[median] internal error"
  with Break -> !med




let aggreg_empty_rel op glist =
  let op_str = MFOTL.string_of_agg_op op in
  let default_value = function
    | Avg -> Float 0.
    | Cnt -> Int 0
    | Min | Max | Sum | Med -> Float 0.
  in
  if glist = [] then
    begin
      (match op with
       | Avg | Med | Min | Max ->
         let err_msg = Printf.sprintf "WARNING: %s applied on empty relation \
                                       at time point %d, timestamp %s! \
                                       Resulting value is 0, by (our) convention.\n"
             op_str !crt_tp (MFOTL.string_of_ts !crt_ts)
         in
         prerr_string err_msg

       | Cnt | Sum -> ()
      );

      Relation.singleton (Tuple.make_tuple [default_value op])
    end
  else
    Relation.empty



let rec add_ext f =
  match f with
  | Pred p ->
    EPred (p, Relation.eval_pred p, Queue.create())

  | Equal (t1, t2) ->
    let rel = Relation.eval_equal t1 t2 in
    ERel rel

  | Neg (Equal (t1, t2)) ->
    let rel = Relation.eval_not_equal t1 t2 in
    ERel rel

  | Neg f -> ENeg (add_ext f)

  | Exists (vl, f1) ->
    let ff1 = add_ext f1 in
    let attr1 = MFOTL.free_vars f1 in
    let pos = List.map (fun v -> Misc.get_pos v attr1) vl in
    let pos = List.sort Pervasives.compare pos in
    let comp = Relation.project_away pos in
    EExists (comp,ff1)

  | Or (f1, f2) ->
    let ff1 = add_ext f1 in
    let ff2 = add_ext f2 in
    let attr1 = MFOTL.free_vars f1 in
    let attr2 = MFOTL.free_vars f2 in
    let comp =
      if attr1 = attr2 then
        Relation.union
      else
        let matches = Table.get_matches attr1 attr2 in
        let new_pos = List.map snd matches in
        (* first reorder rel2 *)
        (fun rel1 rel2 ->
           let rel2' = Relation.reorder new_pos rel2 in
           Relation.union rel1 rel2'
        )
    in
    EOr (comp, ff1, ff2, {arel = None})

  | And (f1, f2) ->
    let attr1 = MFOTL.free_vars f1 in
    let attr2 = MFOTL.free_vars f2 in
    let ff1 = add_ext f1 in
    let f2_is_special = Rewriting.is_special_case attr1 attr2 f2 in
    let ff2 =
      if f2_is_special then ERel Relation.empty
      else match f2 with
        | Neg f2' -> add_ext f2'
        | _ -> add_ext f2
    in
    let comp =
      if f2_is_special then
        if Misc.subset attr2 attr1 then
          let filter_cond = Tuple.get_filter attr1 f2 in
          fun rel1 _ -> Relation.filter filter_cond rel1
        else
          let process_tuple = Tuple.get_tf attr1 f2 in
          fun rel1 _ ->
            Relation.fold
              (fun t res -> Relation.add (process_tuple t) res)
              rel1 Relation.empty
      else
        match f2 with
        | Neg _ ->
          if attr1 = attr2 then
            fun rel1 rel2 -> Relation.diff rel1 rel2
          else
            begin
              assert(Misc.subset attr2 attr1);
              let posl = List.map (fun v -> Misc.get_pos v attr1) attr2 in
              fun rel1 rel2 -> Relation.minus posl rel1 rel2
            end

        | _ ->
          let matches1 = Table.get_matches attr1 attr2 in
          let matches2 = Table.get_matches attr2 attr1 in
          if attr1 = attr2 then
            fun rel1 rel2 -> Relation.inter rel1 rel2
          else if Misc.subset attr1 attr2 then
            fun rel1 rel2 -> Relation.natural_join_sc1 matches2 rel1 rel2
          else if Misc.subset attr2 attr1 then
            fun rel1 rel2 -> Relation.natural_join_sc2 matches1 rel1 rel2
          else
            fun rel1 rel2 -> Relation.natural_join matches1 matches2 rel1 rel2
    in
    EAnd (comp, ff1, ff2, {arel = None})

  | Aggreg (y, (Avg as op), x, glist, Once (intv, f))
  | Aggreg (y, (Sum as op), x, glist, Once (intv, f))
  | Aggreg (y, (Cnt as op), x, glist, Once (intv, f))
  | Aggreg (y, (Med as op), x, glist, Once (intv, f)) ->

    let attr = MFOTL.free_vars f in
    let posx = Misc.get_pos x attr in
    let posG = List.map (fun z -> Misc.get_pos z attr) glist in

    let init_agg_val op cst =
      match op with
      | Med -> Med_aux (1, IntMap.singleton cst 1)
      | _ -> CSA_aux (1, cst)
    in

    let init_agg_values = init_agg_val op in

    let add c xlist =
      try
        let m = IntMap.find c xlist in
        IntMap.add c (m+1) (IntMap.remove c xlist)
      with Not_found -> IntMap.add c 1 xlist
    in

    let remove c xlist =
      let m = IntMap.find c xlist in
      if m = 1 then
        IntMap.remove c xlist
      else
        IntMap.add c (m-1) (IntMap.remove c xlist)
    in

    let update_agg_values add_flag prev_values cst =
      match prev_values with
      | CSA_aux (c, s) ->
        if add_flag
        then true, CSA_aux (c + 1, Predicate.plus s cst)
        else c = 1, CSA_aux (c - 1, Predicate.minus s cst)
      | Med_aux (len, xlist) ->
        if add_flag
        then true, Med_aux (len + 1, add cst xlist)
        else len = 1, Med_aux (len - 1, remove cst xlist)
    in

    let update_state_new state rel =
      let new_rel = ref [] in
      Relation.iter
        (fun tuple ->
           let gtuple = Tuple.projections posG tuple in
           let cst = Tuple.get_at_pos tuple posx in
           new_rel := (tuple, gtuple, cst) :: !new_rel;
           try
             let m = Hashtbl.find state.mset tuple in
             Hashtbl.replace state.mset tuple (m+1);
             assert (m > 0);
             assert (Hashtbl.mem state.hres gtuple)
           with Not_found ->
             Hashtbl.add state.mset tuple 1;
             try
               let agg_values = Hashtbl.find state.hres gtuple in
               let _, new_values = update_agg_values true agg_values cst in
               Hashtbl.replace state.hres gtuple new_values
             with Not_found ->
               Hashtbl.add state.hres gtuple (init_agg_values cst)
        ) rel;
      !new_rel
    in

    let update_state_old state rel =
      List.iter (fun (tuple, gtuple, cst) ->
          let m = Hashtbl.find state.mset tuple in
          assert (m > 0);
          if m = 1 then
            begin
              Hashtbl.remove state.mset tuple;
              let agg_values = Hashtbl.find state.hres gtuple in
              let remove, new_values = update_agg_values false agg_values cst in
              if remove then
                Hashtbl.remove state.hres gtuple
              else
                Hashtbl.replace state.hres gtuple new_values
            end
          else
            Hashtbl.replace state.mset tuple (m-1)
        ) rel
    in

    let get_val_func = function
      | Cnt ->
        (fun x -> match x with
           | CSA_aux (c, _) -> Int c
           | Med_aux _ -> failwith "internal error"
        )
      | Sum ->
        (fun x -> match x with
           | CSA_aux (_, s) -> s
           | Med_aux _ -> failwith "internal error"
        )
      | Avg ->
        (fun x -> match x with
           | CSA_aux (c,s) -> (match s with
               | Int s -> Float ((float_of_int s) /. (float_of_int c))
               | Float s -> Float (s /. (float_of_int c))
               | _ -> failwith "internal error")
           | Med_aux _ -> failwith "internal error"
        )
      | Med ->
        (fun x -> match x with
           | Med_aux (len, xlist) -> median xlist len Predicate.avg
           | CSA_aux _ -> failwith "internal error"
        )

      | _ -> failwith "[add_ext, AggOnce] internal error"
    in

    let get_val = get_val_func op in

    let get_result state =
      if Hashtbl.length state.hres = 0 then
        aggreg_empty_rel op glist
      else
        let res = ref Relation.empty in
        Hashtbl.iter
          (fun gtuple agg_values ->
             res := Relation.add (Tuple.add_first gtuple (get_val agg_values)) !res
          )
          state.hres;
        !res
    in

    let init_state = {
      tw_rels = Queue.create ();
      other_rels = Queue.create ();
      mset = Hashtbl.create 1000;
      hres = Hashtbl.create 100;
    }
    in

    EAggOnce ((add_ext f), intv, init_state, update_state_old, update_state_new, get_result)


  | Aggreg (y, (Min as op), x, glist, Once (intv, f))
  | Aggreg (y, (Max as op), x, glist, Once (intv, f)) ->

    let get_comp_func = function
      | Min -> (fun x y -> - (Pervasives.compare x y))
      | Max -> (fun x y -> Pervasives.compare x y)
      | _ -> failwith "[add_ext, AggMMOnce] internal error"
    in

    (* returns 1 if x better than y, 0 if they are equal, and -1 otherwise *)
    (* for Min: x is better than y iff x < y *)
    (* for Max: x is better than y iff x > y *)
    let is_better = get_comp_func op in

    let attr = MFOTL.free_vars f in
    let posx = Misc.get_pos x attr in
    let posG = List.map (fun z -> Misc.get_pos z attr) glist in

    (* The invariant is:
       if (tsq,v) is before (tsq',v')
       then tsq >= tsq', v' is better or equal than v, and
       we don't have equality in both cases;

       The first condition is ensured by default, as timestamps are
       non-decreasing. We have to enforce the second and third
       consitions. *)
    let rec update_list_new tsq cst dllist =
      if not (Dllist.is_empty dllist) then
        begin
          let (tsq',m) = Dllist.get_first dllist in
          let comp = is_better cst m in
          if comp > 0 then
            begin
              ignore(Dllist.pop_first dllist);
              update_list_new tsq cst dllist
            end
          else if comp = 0 then
            begin
              if tsq <> tsq' then
                begin
                  ignore(Dllist.pop_first dllist);
                  Dllist.add_first (tsq, cst) dllist
                end
                (* else: same element appears previously, no need to
                   update *)
            end
          else
            Dllist.add_first (tsq, cst) dllist
        end
      else
        Dllist.add_first (tsq, cst) dllist
    in

    let update_state_new state tsq rel =
      Relation.iter
        (fun tuple ->
           let gtuple = Tuple.projections posG tuple in
           let cst = Tuple.get_at_pos tuple posx in
           try
             let dllist = Hashtbl.find state.tbl gtuple in
             update_list_new tsq cst dllist
           with Not_found ->
             let dllist = Dllist.singleton (tsq, cst) in
             Hashtbl.add state.tbl gtuple dllist;
        ) rel
    in

    let rec update_list_old tsq dllist =
      if not (Dllist.is_empty dllist) then
        let tsj,_ = Dllist.get_last dllist in
        if not (MFOTL.in_left_ext (MFOTL.ts_minus tsq tsj) intv) then
          begin
            ignore(Dllist.pop_last dllist);
            update_list_old tsq dllist
          end
    in

    let update_state_old state tsq =
      Hashtbl.iter (fun gtuple dllist ->
          update_list_old tsq dllist;
          if Dllist.is_empty dllist then
            (* TODO: is it safe to modify the hash table while we iterate on it!? *)
            Hashtbl.remove state.tbl gtuple;
        ) state.tbl
    in

    let get_result state =
      if Hashtbl.length state.tbl = 0 then
        aggreg_empty_rel op glist
      else
        let res = ref Relation.empty in
        Hashtbl.iter
          (fun gtuple dllist ->
             let _, agg_val = Dllist.get_last dllist in
             res := Relation.add (Tuple.add_first gtuple agg_val) !res
          )
          state.tbl;
        !res
    in

    let init_state = {
      non_tw_rels = Queue.create ();
      tbl = Hashtbl.create 100;
    }
    in

    EAggMMOnce ((add_ext f), intv, init_state, update_state_old, update_state_new, get_result)



  | Aggreg (y, Avg, x, glist, f) ->
    let attr = MFOTL.free_vars f in
    let posx = Misc.get_pos x attr in
    let posG = List.map (fun z -> Misc.get_pos z attr) glist in
    let init_value = fun v -> (v,1) in
    let update =
      fun (os,oc) nv ->
        match os, nv with
        | Int s, Int v -> (Int (s + v), oc + 1)
        | Float s, Float v -> (Float (s +. v), oc + 1)
        | _ -> failwith "[Aggreg.Avg, update] internal error"
    in
    let comp_map = comp_aggreg init_value update posx posG in
    let comp rel =
      if Relation.is_empty rel then
        aggreg_empty_rel Avg glist
      else
        let map = comp_map rel in
        let new_rel = ref Relation.empty in
        Hashtbl.iter (fun tuple (s,c) ->
            let s' = match s with
              | Float x -> x
              | Int x -> float_of_int x
              | _ -> failwith "[Aggreg.Avg, comp] internal error"
            in
            new_rel := Relation.add
                (Tuple.add_first tuple (Float (s' /. (float_of_int c)))) !new_rel;
          ) map;
        !new_rel
    in
    EAggreg (comp, add_ext f)

  | Aggreg (y, Med, x, glist, f) ->
    let attr = MFOTL.free_vars f in
    let posx = Misc.get_pos x attr in
    let posG = List.map (fun z -> Misc.get_pos z attr) glist in
    let init_value = fun v -> (1, [v]) in
    let update = fun (len, old_list) nv -> (len+1, nv::old_list) in
    let comp_map = comp_aggreg init_value update posx posG in
    let fmed a b =
      match a, b with
      | Int x, Int y -> Int ((x+y)/2)
      | Float x, Float y -> Float ((x+.y)/.2.)
      | _ -> failwith "[add_ext] type error"
    in
    let comp rel =
      if Relation.is_empty rel then
        aggreg_empty_rel Med glist
      else
        let map = comp_map rel in
        let new_rel = ref Relation.empty in
        Hashtbl.iter (fun tuple (len, vlist) ->
            let vlist = List.sort Pervasives.compare vlist in
            let med = Misc.median vlist len fmed in
            new_rel := Relation.add (Tuple.add_first tuple med) !new_rel;
          ) map;
        !new_rel
    in
    EAggreg (comp, add_ext f)

  | Aggreg (y, op, x, glist, f) ->
    let attr = MFOTL.free_vars f in
    let posx = Misc.get_pos x attr in
    let posG = List.map (fun z -> Misc.get_pos z attr) glist in
    let init_value, update =
      match op with
      | Cnt -> (fun v -> Int 1),
               (fun ov _ -> match ov with
                  | Int ov -> Int (ov + 1)
                  | _ -> failwith "[add_ext, Aggreg] internal error")
      | Min -> (fun v -> v), (fun ov nv -> min ov nv)
      | Max -> (fun v -> v), (fun ov nv -> max ov nv)
      | Sum -> (fun v -> v), (fun ov nv -> match ov, nv with
          | (Int ov), (Int nv) -> Int (ov + nv)
          | (Float ov), (Float nv) -> Float (ov +. nv)
          | _ -> failwith "[add_ext, Aggreg] internal error")
      | Avg | Med -> failwith "[add_ext, Aggreg] internal error"
    in
    let comp_map = comp_aggreg init_value update posx posG in
    let comp rel =
      if Relation.is_empty rel then
        aggreg_empty_rel op glist
      else
        let map = comp_map rel in
        let new_rel = ref Relation.empty in
        Hashtbl.iter (fun tuple v ->
            new_rel := Relation.add (Tuple.add_first tuple v) !new_rel;
          ) map;
        !new_rel
    in
    EAggreg (comp, add_ext f)

  | Prev (intv, f) ->
    let ff = add_ext f in
    EPrev (intv, ff, {ptsq = MFOTL.ts_invalid})

  | Next (intv, f) ->
    let ff = add_ext f in
    ENext (intv, ff, {init = true})

  | Since (intv,f1,f2) ->
    let attr1 = MFOTL.free_vars f1 in
    let attr2 = MFOTL.free_vars f2 in
    let ef1, neg =
      (match f1 with
       | Neg f1' -> f1',true
       | _ -> f1,false
      )
    in
    let comp =
      if neg then
        let posl = List.map (fun v -> Misc.get_pos v attr2) attr1 in
        assert(Misc.subset attr1 attr2);
        fun relj rel1 -> Relation.minus posl relj rel1
      else
        let matches2 = Table.get_matches attr2 attr1 in
        fun relj rel1 -> Relation.natural_join_sc2 matches2 relj rel1
    in
    let ff1 = add_ext ef1 in
    let ff2 = add_ext f2 in
    if snd intv = Inf then
      let inf = {sres = Relation.empty; sarel2 = None; saauxrels = Mqueue.create()} in
      ESinceA (comp,intv,ff1,ff2,inf)
    else
      let inf = {srel2 = None; sauxrels = Mqueue.create()} in
      ESince (comp,intv,ff1,ff2,inf)

  | Once ((_, Inf) as intv, f) ->
    let ff = add_ext f in
    EOnceA (intv,ff,{ores = Relation.empty;
                     oaauxrels = Mqueue.create()})

  | Once (intv,f) ->
    let ff = add_ext f in
    if fst intv = CBnd MFOTL.ts_null then
      EOnceZ (intv,ff,{oztree = LNode {l = -1;
                                       r = -1;
                                       res = Some (Relation.empty)};
                       ozlast = Dllist.void;
                       ozauxrels = Dllist.empty()})
    else
      EOnce (intv,ff,{otree = LNode {l = MFOTL.ts_invalid;
                                     r = MFOTL.ts_invalid;
                                     res = Some (Relation.empty)};
                      olast = Dllist.void;
                      oauxrels = Dllist.empty()})

  | Until (intv,f1,f2) ->
    let attr1 = MFOTL.free_vars f1 in
    let attr2 = MFOTL.free_vars f2 in
    let ef1, neg =
      (match f1 with
       | Neg f1' -> f1',true
       | _ -> f1,false
      )
    in
    let ff1 = add_ext ef1 in
    let ff2 = add_ext f2 in
    if neg then
      let comp =
        let posl = List.map (fun v -> Misc.get_pos v attr2) attr1 in
        assert(Misc.subset attr1 attr2);
        fun relj rel1 -> Relation.minus posl relj rel1
      in
      let inf = {
        last1 = NEval.void;
        last2 = NEval.void;
        listrel1 = Dllist.empty();
        listrel2 = Dllist.empty()}
      in
      ENUntil (comp,intv,ff1,ff2,inf)
    else
      let comp =
        let matches2 = Table.get_matches attr2 attr1 in
        fun relj rel1 -> Relation.natural_join_sc2 matches2 relj rel1
      in
      let inf = {ulast = NEval.void;
                 ufirst = false;
                 ures = Relation.empty;
                 urel2 = None;
                 raux = Sj.empty();
                 saux = Sk.empty()}
      in
      EUntil (comp,intv,ff1,ff2,inf)


  | Eventually (intv,f) ->
    let ff = add_ext f in
    if fst intv = CBnd MFOTL.ts_null then
      EEventuallyZ (intv,ff,{eztree = LNode {l = -1;
                                             r = -1;
                                             res = Some (Relation.empty)};
                             ezlast = Dllist.void;
                             ezlastev = NEval.void;
                             ezauxrels = Dllist.empty()})
    else
      EEventually (intv,ff,{etree = LNode {l = MFOTL.ts_invalid;
                                           r = MFOTL.ts_invalid;
                                           res = Some (Relation.empty)};
                            elast = Dllist.void;
                            elastev = NEval.void;
                            eauxrels = Dllist.empty()})

  | _ -> failwith "[add_ext] internal error"




let resumefile = ref ""
let dumpfile = ref ""
let lastts = ref MFOTL.ts_invalid

(* The arguments are:
   lexbuf - the lexer buffer (holds current state of the scanner)
   ff - the extended MFOTL formula
   closed - true iff [ff] is a ground formula
   neval - the queue of no-yet evaluted indexes/entries
   i - the index of current entry in the log file
   ([i] may be different from the current time point when
   filter_empty_tp is enabled)
*)
let check_log lexbuf ff closed neval i =
  let finish () =
    if Misc.debugging Dbg_perf then
      Perf.check_log_end i !lastts
  in
  let rec loop ffl i =
    if Misc.debugging Dbg_perf then
      Perf.check_log i !lastts;
    match Log.get_next_entry lexbuf with
    | Some (tp, ts, db) ->
      if ts >= !lastts then
        begin
          crt_tp := tp;
          crt_ts := ts;
          add_index ff tp ts db;
          NEval.add_last (tp, ts) neval;
          let cont = process_index ff closed neval tp in
          lastts := ts;
          if cont then
            loop ffl (i + 1)
          else
            finish ()
        end
      else
      if !Misc.stop_at_out_of_order_ts then
        let msg = Printf.sprintf "[Algorithm.check_log] Error: OUT OF ORDER TIMESTAMP: %s \
                                  (last_ts: %s)" (MFOTL.string_of_ts ts) (MFOTL.string_of_ts !lastts) in
        failwith msg
      else
        begin
          Printf.eprintf "[Algorithm.check_log] skipping OUT OF ORDER TIMESTAMP: %s \
                          (last_ts: %s)\n%!"
            (MFOTL.string_of_ts ts) (MFOTL.string_of_ts !lastts);
          loop ffl i
        end

    | None -> finish ()
  in
  loop ff i







let monitor_lexbuf lexbuf f =
  check_log lexbuf (add_ext f) (MFOTL.free_vars f = []) (NEval.empty()) 0

let monitor_string log f =
  (let lexbuf = Lexing.from_string log in
   lastts := MFOTL.ts_invalid;
   crt_tp := -1;
   crt_ts := MFOTL.ts_invalid;
   Log.tp := 0;
   Log.skipped_tps := 0;
   Log.last := false;
   monitor_lexbuf lexbuf f;
   Lexing.flush_input lexbuf;)

let monitor logfile =
  let lexbuf = Log.log_open logfile in
  monitor_lexbuf lexbuf


let test_filter logfile f =
  let lexbuf = Log.log_open logfile in
  let rec loop f i =
    match Log.get_next_entry lexbuf with
    | Some (tp,ts,db) ->
      loop f tp
    | None ->
      Printf.printf "end of log, processed %d time points\n" (i - 1)
  in
  loop f 0
