(set-logic LIA)

(define-fun __node_init_intloopcounter_0 ((intloopcounter.usr.x_a_0 Bool) (intloopcounter.usr.out_a_0 Bool) (intloopcounter.res.init_flag_a_0 Bool) (intloopcounter.impl.usr.time_a_0 Int)) Bool
    (and (and (= intloopcounter.impl.usr.time_a_0 0) (= intloopcounter.usr.out_a_0 (= intloopcounter.impl.usr.time_a_0 2))) intloopcounter.res.init_flag_a_0))
(define-fun __node_trans_intloopcounter_0 ((intloopcounter.usr.x_a_1 Bool) (intloopcounter.usr.out_a_1 Bool) (intloopcounter.res.init_flag_a_1 Bool) (intloopcounter.impl.usr.time_a_1 Int) (intloopcounter.usr.x_a_0 Bool) (intloopcounter.usr.out_a_0 Bool) (intloopcounter.res.init_flag_a_0 Bool) (intloopcounter.impl.usr.time_a_0 Int)) Bool
    (and (and (= intloopcounter.impl.usr.time_a_1 (ite (= intloopcounter.impl.usr.time_a_0 3) 0 (+ intloopcounter.impl.usr.time_a_0 1))) (= intloopcounter.usr.out_a_1 (= intloopcounter.impl.usr.time_a_1 2))) (not intloopcounter.res.init_flag_a_1)))
(define-fun __node_init_greycounter_0 ((greycounter.usr.x_a_0 Bool) (greycounter.usr.out_a_0 Bool) (greycounter.res.init_flag_a_0 Bool) (greycounter.impl.usr.a_a_0 Bool) (greycounter.impl.usr.b_a_0 Bool)) Bool
    (and (and (and (= greycounter.impl.usr.b_a_0 false) (= greycounter.impl.usr.a_a_0 false)) (= greycounter.usr.out_a_0 (and greycounter.impl.usr.a_a_0 greycounter.impl.usr.b_a_0))) greycounter.res.init_flag_a_0))
(define-fun __node_trans_greycounter_0 ((greycounter.usr.x_a_1 Bool) (greycounter.usr.out_a_1 Bool) (greycounter.res.init_flag_a_1 Bool) (greycounter.impl.usr.a_a_1 Bool) (greycounter.impl.usr.b_a_1 Bool) (greycounter.usr.x_a_0 Bool) (greycounter.usr.out_a_0 Bool) (greycounter.res.init_flag_a_0 Bool) (greycounter.impl.usr.a_a_0 Bool) (greycounter.impl.usr.b_a_0 Bool)) Bool
    (and (and (and (= greycounter.impl.usr.b_a_1 greycounter.impl.usr.a_a_0) (= greycounter.impl.usr.a_a_1 (not greycounter.impl.usr.b_a_0))) (= greycounter.usr.out_a_1 (and greycounter.impl.usr.a_a_1 greycounter.impl.usr.b_a_1))) (not greycounter.res.init_flag_a_1)))
(define-fun __node_init_top_0 ((top.usr.x_a_0 Bool) (top.usr.OK_a_0 Bool) (top.res.init_flag_a_0 Bool) (top.res.abs_0_a_0 Bool) (top.res.abs_1_a_0 Bool) (top.res.inst_4_a_0 Bool) (top.res.inst_3_a_0 Bool) (top.res.inst_2_a_0 Bool) (top.res.inst_1_a_0 Bool) (top.res.inst_0_a_0 Int)) Bool
    (let ((X1 Bool top.res.abs_1_a_0)) (let ((X2 Bool top.res.abs_0_a_0)) (and (and (and (= top.usr.OK_a_0 (= X2 X1)) (__node_init_greycounter_0 top.usr.x_a_0 top.res.abs_0_a_0 top.res.inst_4_a_0 top.res.inst_3_a_0 top.res.inst_2_a_0)) (__node_init_intloopcounter_0 top.usr.x_a_0 top.res.abs_1_a_0 top.res.inst_1_a_0 top.res.inst_0_a_0)) top.res.init_flag_a_0))))
(define-fun __node_trans_top_0 ((top.usr.x_a_1 Bool) (top.usr.OK_a_1 Bool) (top.res.init_flag_a_1 Bool) (top.res.abs_0_a_1 Bool) (top.res.abs_1_a_1 Bool) (top.res.inst_4_a_1 Bool) (top.res.inst_3_a_1 Bool) (top.res.inst_2_a_1 Bool) (top.res.inst_1_a_1 Bool) (top.res.inst_0_a_1 Int) (top.usr.x_a_0 Bool) (top.usr.OK_a_0 Bool) (top.res.init_flag_a_0 Bool) (top.res.abs_0_a_0 Bool) (top.res.abs_1_a_0 Bool) (top.res.inst_4_a_0 Bool) (top.res.inst_3_a_0 Bool) (top.res.inst_2_a_0 Bool) (top.res.inst_1_a_0 Bool) (top.res.inst_0_a_0 Int)) Bool
    (let ((X1 Bool top.res.abs_1_a_1)) (let ((X2 Bool top.res.abs_0_a_1)) (and (and (and (= top.usr.OK_a_1 (= X2 X1)) (__node_trans_greycounter_0 top.usr.x_a_1 top.res.abs_0_a_1 top.res.inst_4_a_1 top.res.inst_3_a_1 top.res.inst_2_a_1 top.usr.x_a_0 top.res.abs_0_a_0 top.res.inst_4_a_0 top.res.inst_3_a_0 top.res.inst_2_a_0)) (__node_trans_intloopcounter_0 top.usr.x_a_1 top.res.abs_1_a_1 top.res.inst_1_a_1 top.res.inst_0_a_1 top.usr.x_a_0 top.res.abs_1_a_0 top.res.inst_1_a_0 top.res.inst_0_a_0)) (not top.res.init_flag_a_1)))))
(synth-inv str_invariant ((top.usr.x Bool) (top.usr.OK Bool) (top.res.init_flag Bool) (top.res.abs_0 Bool) (top.res.abs_1 Bool) (top.res.inst_4 Bool) (top.res.inst_3 Bool) (top.res.inst_2 Bool) (top.res.inst_1 Bool) (top.res.inst_0 Int)))

(declare-primed-var top.usr.x Bool)
(declare-primed-var top.usr.OK Bool)
(declare-primed-var top.res.init_flag Bool)
(declare-primed-var top.res.abs_0 Bool)
(declare-primed-var top.res.abs_1 Bool)
(declare-primed-var top.res.inst_4 Bool)
(declare-primed-var top.res.inst_3 Bool)
(declare-primed-var top.res.inst_2 Bool)
(declare-primed-var top.res.inst_1 Bool)
(declare-primed-var top.res.inst_0 Int)
(define-fun init ((top.usr.x Bool) (top.usr.OK Bool) (top.res.init_flag Bool) (top.res.abs_0 Bool) (top.res.abs_1 Bool) (top.res.inst_4 Bool) (top.res.inst_3 Bool) (top.res.inst_2 Bool) (top.res.inst_1 Bool) (top.res.inst_0 Int)) Bool
    (let ((X1 Bool top.res.abs_1)) (let ((X2 Bool top.res.abs_0)) (and (and (and (= top.usr.OK (= X2 X1)) (__node_init_greycounter_0 top.usr.x top.res.abs_0 top.res.inst_4 top.res.inst_3 top.res.inst_2)) (__node_init_intloopcounter_0 top.usr.x top.res.abs_1 top.res.inst_1 top.res.inst_0)) top.res.init_flag))))
(define-fun trans ((top.usr.x Bool) (top.usr.OK Bool) (top.res.init_flag Bool) (top.res.abs_0 Bool) (top.res.abs_1 Bool) (top.res.inst_4 Bool) (top.res.inst_3 Bool) (top.res.inst_2 Bool) (top.res.inst_1 Bool) (top.res.inst_0 Int) (top.usr.x! Bool) (top.usr.OK! Bool) (top.res.init_flag! Bool) (top.res.abs_0! Bool) (top.res.abs_1! Bool) (top.res.inst_4! Bool) (top.res.inst_3! Bool) (top.res.inst_2! Bool) (top.res.inst_1! Bool) (top.res.inst_0! Int)) Bool
    (let ((X1 Bool top.res.abs_1!)) (let ((X2 Bool top.res.abs_0!)) (and (and (and (= top.usr.OK! (= X2 X1)) (__node_trans_greycounter_0 top.usr.x! top.res.abs_0! top.res.inst_4! top.res.inst_3! top.res.inst_2! top.usr.x top.res.abs_0 top.res.inst_4 top.res.inst_3 top.res.inst_2)) (__node_trans_intloopcounter_0 top.usr.x! top.res.abs_1! top.res.inst_1! top.res.inst_0! top.usr.x top.res.abs_1 top.res.inst_1 top.res.inst_0)) (not top.res.init_flag!)))))
(define-fun prop ((top.usr.x Bool) (top.usr.OK Bool) (top.res.init_flag Bool) (top.res.abs_0 Bool) (top.res.abs_1 Bool) (top.res.inst_4 Bool) (top.res.inst_3 Bool) (top.res.inst_2 Bool) (top.res.inst_1 Bool) (top.res.inst_0 Int)) Bool
    top.usr.OK)

(inv-constraint str_invariant init trans prop)

(check-synth)

