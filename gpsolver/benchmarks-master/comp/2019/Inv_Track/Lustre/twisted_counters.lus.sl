(set-logic LIA)

(define-fun __node_init_intloop6counter_0 ((intloop6counter.usr.x_a_0 Bool) (intloop6counter.usr.out_a_0 Bool) (intloop6counter.res.init_flag_a_0 Bool) (intloop6counter.impl.usr.time_a_0 Int)) Bool
    (and (and (= intloop6counter.impl.usr.time_a_0 0) (= intloop6counter.usr.out_a_0 (= intloop6counter.impl.usr.time_a_0 5))) intloop6counter.res.init_flag_a_0))
(define-fun __node_trans_intloop6counter_0 ((intloop6counter.usr.x_a_1 Bool) (intloop6counter.usr.out_a_1 Bool) (intloop6counter.res.init_flag_a_1 Bool) (intloop6counter.impl.usr.time_a_1 Int) (intloop6counter.usr.x_a_0 Bool) (intloop6counter.usr.out_a_0 Bool) (intloop6counter.res.init_flag_a_0 Bool) (intloop6counter.impl.usr.time_a_0 Int)) Bool
    (and (and (= intloop6counter.impl.usr.time_a_1 (ite (= intloop6counter.impl.usr.time_a_0 5) 2 (ite (= intloop6counter.impl.usr.time_a_0 4) (ite (not intloop6counter.usr.x_a_0) 3 5) (+ intloop6counter.impl.usr.time_a_0 1)))) (= intloop6counter.usr.out_a_1 (= intloop6counter.impl.usr.time_a_1 5))) (not intloop6counter.res.init_flag_a_1)))
(define-fun __node_init_loop6counter_0 ((loop6counter.usr.x_a_0 Bool) (loop6counter.usr.out_a_0 Bool) (loop6counter.res.init_flag_a_0 Bool) (loop6counter.impl.usr.a_a_0 Bool) (loop6counter.impl.usr.b_a_0 Bool) (loop6counter.impl.usr.c_a_0 Bool)) Bool
    (and (and (and (and (= loop6counter.impl.usr.c_a_0 false) (= loop6counter.impl.usr.a_a_0 false)) (= loop6counter.usr.out_a_0 (and loop6counter.impl.usr.a_a_0 loop6counter.impl.usr.c_a_0))) (= loop6counter.impl.usr.b_a_0 false)) loop6counter.res.init_flag_a_0))
(define-fun __node_trans_loop6counter_0 ((loop6counter.usr.x_a_1 Bool) (loop6counter.usr.out_a_1 Bool) (loop6counter.res.init_flag_a_1 Bool) (loop6counter.impl.usr.a_a_1 Bool) (loop6counter.impl.usr.b_a_1 Bool) (loop6counter.impl.usr.c_a_1 Bool) (loop6counter.usr.x_a_0 Bool) (loop6counter.usr.out_a_0 Bool) (loop6counter.res.init_flag_a_0 Bool) (loop6counter.impl.usr.a_a_0 Bool) (loop6counter.impl.usr.b_a_0 Bool) (loop6counter.impl.usr.c_a_0 Bool)) Bool
    (and (and (and (and (= loop6counter.impl.usr.c_a_1 (not loop6counter.impl.usr.c_a_0)) (= loop6counter.impl.usr.a_a_1 (or (and loop6counter.impl.usr.b_a_0 loop6counter.impl.usr.c_a_0) (and (and loop6counter.usr.x_a_0 loop6counter.impl.usr.a_a_0) (not loop6counter.impl.usr.c_a_0))))) (= loop6counter.usr.out_a_1 (and loop6counter.impl.usr.a_a_1 loop6counter.impl.usr.c_a_1))) (= loop6counter.impl.usr.b_a_1 (or (or (and (not loop6counter.impl.usr.b_a_0) loop6counter.impl.usr.c_a_0) (and loop6counter.impl.usr.b_a_0 (not loop6counter.impl.usr.c_a_0))) (and (not loop6counter.usr.x_a_0) loop6counter.impl.usr.a_a_0)))) (not loop6counter.res.init_flag_a_1)))
(define-fun __node_init_top_0 ((top.usr.x_a_0 Bool) (top.usr.OK_a_0 Bool) (top.res.init_flag_a_0 Bool) (top.res.abs_0_a_0 Bool) (top.res.abs_1_a_0 Bool) (top.res.inst_5_a_0 Bool) (top.res.inst_4_a_0 Bool) (top.res.inst_3_a_0 Bool) (top.res.inst_2_a_0 Bool) (top.res.inst_1_a_0 Bool) (top.res.inst_0_a_0 Int)) Bool
    (let ((X1 Bool top.res.abs_1_a_0)) (let ((X2 Bool top.res.abs_0_a_0)) (and (and (and (= top.usr.OK_a_0 (or (not top.usr.x_a_0) (= X2 X1))) (__node_init_loop6counter_0 top.usr.x_a_0 top.res.abs_0_a_0 top.res.inst_5_a_0 top.res.inst_4_a_0 top.res.inst_3_a_0 top.res.inst_2_a_0)) (__node_init_intloop6counter_0 top.usr.x_a_0 top.res.abs_1_a_0 top.res.inst_1_a_0 top.res.inst_0_a_0)) top.res.init_flag_a_0))))
(define-fun __node_trans_top_0 ((top.usr.x_a_1 Bool) (top.usr.OK_a_1 Bool) (top.res.init_flag_a_1 Bool) (top.res.abs_0_a_1 Bool) (top.res.abs_1_a_1 Bool) (top.res.inst_5_a_1 Bool) (top.res.inst_4_a_1 Bool) (top.res.inst_3_a_1 Bool) (top.res.inst_2_a_1 Bool) (top.res.inst_1_a_1 Bool) (top.res.inst_0_a_1 Int) (top.usr.x_a_0 Bool) (top.usr.OK_a_0 Bool) (top.res.init_flag_a_0 Bool) (top.res.abs_0_a_0 Bool) (top.res.abs_1_a_0 Bool) (top.res.inst_5_a_0 Bool) (top.res.inst_4_a_0 Bool) (top.res.inst_3_a_0 Bool) (top.res.inst_2_a_0 Bool) (top.res.inst_1_a_0 Bool) (top.res.inst_0_a_0 Int)) Bool
    (let ((X1 Bool top.res.abs_1_a_1)) (let ((X2 Bool top.res.abs_0_a_1)) (and (and (and (= top.usr.OK_a_1 (or (not top.usr.x_a_1) (= X2 X1))) (__node_trans_loop6counter_0 top.usr.x_a_1 top.res.abs_0_a_1 top.res.inst_5_a_1 top.res.inst_4_a_1 top.res.inst_3_a_1 top.res.inst_2_a_1 top.usr.x_a_0 top.res.abs_0_a_0 top.res.inst_5_a_0 top.res.inst_4_a_0 top.res.inst_3_a_0 top.res.inst_2_a_0)) (__node_trans_intloop6counter_0 top.usr.x_a_1 top.res.abs_1_a_1 top.res.inst_1_a_1 top.res.inst_0_a_1 top.usr.x_a_0 top.res.abs_1_a_0 top.res.inst_1_a_0 top.res.inst_0_a_0)) (not top.res.init_flag_a_1)))))
(synth-inv str_invariant ((top.usr.x Bool) (top.usr.OK Bool) (top.res.init_flag Bool) (top.res.abs_0 Bool) (top.res.abs_1 Bool) (top.res.inst_5 Bool) (top.res.inst_4 Bool) (top.res.inst_3 Bool) (top.res.inst_2 Bool) (top.res.inst_1 Bool) (top.res.inst_0 Int)))

(declare-primed-var top.usr.x Bool)
(declare-primed-var top.usr.OK Bool)
(declare-primed-var top.res.init_flag Bool)
(declare-primed-var top.res.abs_0 Bool)
(declare-primed-var top.res.abs_1 Bool)
(declare-primed-var top.res.inst_5 Bool)
(declare-primed-var top.res.inst_4 Bool)
(declare-primed-var top.res.inst_3 Bool)
(declare-primed-var top.res.inst_2 Bool)
(declare-primed-var top.res.inst_1 Bool)
(declare-primed-var top.res.inst_0 Int)
(define-fun init ((top.usr.x Bool) (top.usr.OK Bool) (top.res.init_flag Bool) (top.res.abs_0 Bool) (top.res.abs_1 Bool) (top.res.inst_5 Bool) (top.res.inst_4 Bool) (top.res.inst_3 Bool) (top.res.inst_2 Bool) (top.res.inst_1 Bool) (top.res.inst_0 Int)) Bool
    (let ((X1 Bool top.res.abs_1)) (let ((X2 Bool top.res.abs_0)) (and (and (and (= top.usr.OK (or (not top.usr.x) (= X2 X1))) (__node_init_loop6counter_0 top.usr.x top.res.abs_0 top.res.inst_5 top.res.inst_4 top.res.inst_3 top.res.inst_2)) (__node_init_intloop6counter_0 top.usr.x top.res.abs_1 top.res.inst_1 top.res.inst_0)) top.res.init_flag))))
(define-fun trans ((top.usr.x Bool) (top.usr.OK Bool) (top.res.init_flag Bool) (top.res.abs_0 Bool) (top.res.abs_1 Bool) (top.res.inst_5 Bool) (top.res.inst_4 Bool) (top.res.inst_3 Bool) (top.res.inst_2 Bool) (top.res.inst_1 Bool) (top.res.inst_0 Int) (top.usr.x! Bool) (top.usr.OK! Bool) (top.res.init_flag! Bool) (top.res.abs_0! Bool) (top.res.abs_1! Bool) (top.res.inst_5! Bool) (top.res.inst_4! Bool) (top.res.inst_3! Bool) (top.res.inst_2! Bool) (top.res.inst_1! Bool) (top.res.inst_0! Int)) Bool
    (let ((X1 Bool top.res.abs_1!)) (let ((X2 Bool top.res.abs_0!)) (and (and (and (= top.usr.OK! (or (not top.usr.x!) (= X2 X1))) (__node_trans_loop6counter_0 top.usr.x! top.res.abs_0! top.res.inst_5! top.res.inst_4! top.res.inst_3! top.res.inst_2! top.usr.x top.res.abs_0 top.res.inst_5 top.res.inst_4 top.res.inst_3 top.res.inst_2)) (__node_trans_intloop6counter_0 top.usr.x! top.res.abs_1! top.res.inst_1! top.res.inst_0! top.usr.x top.res.abs_1 top.res.inst_1 top.res.inst_0)) (not top.res.init_flag!)))))
(define-fun prop ((top.usr.x Bool) (top.usr.OK Bool) (top.res.init_flag Bool) (top.res.abs_0 Bool) (top.res.abs_1 Bool) (top.res.inst_5 Bool) (top.res.inst_4 Bool) (top.res.inst_3 Bool) (top.res.inst_2 Bool) (top.res.inst_1 Bool) (top.res.inst_0 Int)) Bool
    top.usr.OK)

(inv-constraint str_invariant init trans prop)

(check-synth)

