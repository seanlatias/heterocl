/*!
 *  Copyright (c) 2020 by Contributors
 * \file inject_monitor.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <arithmetic/Substitute.h>

namespace TVM {
namespace ir {

namespace {

}

class CodeInjector final : public IRMutator {
 public:
  CodeInjector(VarExpr& counter) : counter_(counter) {}

  Stmt Mutate_(const For* op, const Stmt& s) {
    Stmt pre_stmt = Evaluate::make(0);
    // print any accesses before entering a loop
    if (accesses_.find(current_id_) != accesses_.end() &&
        accesses_[current_id_].size() > 0) {
      for (size_t i = 0; i < current_loops_.size(); i++) {
        pre_stmt = Block::make(
            Print::make({UIntImm::make(UInt(32), current_loops_[i]), current_loopvars_[i]}, "Loop2 (%d, %d)\n"),
            pre_stmt);
      }
      for (size_t i = 0; i < accesses_[current_id_].size(); i++) {
        pre_stmt = Block::make(pre_stmt, accesses_[current_id_][i]);
      }
      accesses_[current_id_].clear();
    }

    // preparing for entering a loop
    loop_id_ += 1;
    current_loops_.push_back(loop_id_);
    current_loopvars_.push_back(op->loop_var);
    size_t previous_id = current_id_;
    current_id_ = loop_id_;
    Stmt incr_counter = Store::make(
        counter_, 
        Load::make(UInt(32), counter_, 0, const_true(1)) + 1, 
        0, const_true(1));
    Stmt print_stmt = Print::make({Load::make(UInt(32), counter_, 0, const_true(1))}, "@%d\n");
    print_stmt = Block::make(incr_counter, print_stmt);
    Stmt body = this->Mutate(op->body);
    if (accesses_.find(current_id_) != accesses_.end() &&
        accesses_[current_id_].size() > 0) {
      for (size_t i = 0; i < current_loops_.size(); i++) {
        body = Block::make(
            Print::make({UIntImm::make(UInt(32), current_loops_[i]), current_loopvars_[i]}, "Loop2 (%d, %d)\n"),
            body);
      }
      for (size_t i = 0; i < accesses_[current_id_].size(); i++) {
        body = Block::make(body, accesses_[current_id_][i]);
      }
    }
    body = Block::make(print_stmt, body);
    current_id_ = previous_id;
    current_loops_.pop_back();
    current_loopvars_.pop_back();
    Stmt ret =  For::make(op->loop_var, op->min, op->extent, op->for_type,
                          op->device_api, body, op->annotate_keys, op->annotate_values);
    return Block::make(pre_stmt, ret);
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    std::string name = op->buffer_var->name_hint;
    name = "Load (" + name + ", %d)\n";
    Stmt print_stmt = Print::make({op->index}, name);
    accesses_[current_id_].push_back(print_stmt);
    load_collector_[op->buffer_var.get()].push_back(op->index);
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    std::string name = op->buffer_var->name_hint;
    name = "Store (" + name + ", %d)\n";
    Stmt print_stmt = Print::make({op->index}, name);
    accesses_[current_id_].push_back(print_stmt);
    store_collector_[op->buffer_var.get()].push_back(op->index);
    return IRMutator::Mutate_(op, s);
  }

 private:
  VarExpr& counter_;
  size_t loop_id_{0};
  size_t current_id_;
  std::map<size_t, std::vector<Stmt> > accesses_;
  std::vector<size_t> current_loops_;
  std::vector<VarExpr> current_loopvars_;
  std::unordered_map<const Variable*, std::vector<Expr>> load_collector_;
  std::unordered_map<const Variable*, std::vector<Expr>> store_collector_;
};

Stmt InjectMonitor(Stmt stmt) {
  VarExpr counter = VarExpr("_monitor_counter");
  stmt = CodeInjector(counter).Mutate(stmt);
  stmt = Block::make(Store::make(counter, 0, 0, const_true(1)), stmt);
  stmt = Allocate::make(counter, UInt(32), {1}, const_true(1), stmt);
  return stmt;
}

} // end namespace ir
} // end namespace TVM
