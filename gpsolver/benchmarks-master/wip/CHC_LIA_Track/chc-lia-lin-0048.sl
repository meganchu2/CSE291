(set-logic CHC_LIA)

(synth-fun state ((x_0 Bool) (x_1 Bool) (x_2 Bool) (x_3 Bool) (x_4 Int) (x_5 Int) (x_6 Int) (x_7 Int)) Bool)

(constraint (forall ((.s.1 Bool) (.s.0 Bool) (.s.2 Bool) (.s.3 Bool) (main.i Int) (main.j Int) (main.k Int) (main.n Int)) (=> (and (not .s.1) .s.0 (not .s.2) (not .s.3)) (state .s.1 .s.0 .s.2 .s.3 main.i main.j main.k main.n))))
(constraint (forall ((.s.1 Bool) (.s.0 Bool) (.s.2 Bool) (.s.3 Bool) (main.i Int) (main.j Int) (main.k Int) (main.n Int) (.s.0.next Bool) (.s.1.next Bool) (.s.2.next Bool) (.s.3.next Bool) (main.i.next Int) (main.j.next Int) (main.k.next Int) (main.n.next Int)) (let ((a!1 (not (and .s.3 (and (not .s.2) (and .s.1 .s.0))))) (a!2 (and (and (and .s.0.next .s.1.next) (not .s.2.next)) .s.3.next (= main.i main.i.next) (= main.j main.j.next) (= main.k main.k.next) (= main.n main.n.next))) (a!3 (and .s.3 (and (not .s.2) (and .s.1 (not .s.0))))) (a!4 (and (= main.n main.n.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.3.next) (and (not .s.2.next) (and (not .s.1.next) (not .s.0.next))))) (a!5 (and (and (and (not .s.1) .s.0) (not .s.2)) .s.3)) (a!6 (and (= main.n main.n.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.3.next (and (not .s.2.next) (and .s.1.next (not .s.0.next))))) (a!7 (and .s.3 (and (not .s.2) (and (not .s.1) (not .s.0))))) (a!10 (and (= main.n main.n.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.3.next) .s.2.next (and (not .s.1.next) (not .s.0.next)))) (a!12 (not (and (not .s.3) .s.2 (and .s.1 (not .s.0))))) (a!13 (= (+ main.j (* (- 1) main.j.next)) (- 1))) (a!14 (and (<= 0 main.i) (and (not .s.3) (and (not .s.1) .s.0) .s.2))) (a!16 (and (not (<= 0 main.i)) (and (not .s.3) (and (not .s.1) .s.0) .s.2))) (a!17 (and (and (not .s.3) .s.2 (and (not .s.1) (not .s.0))) (not (<= main.j main.n)))) (a!18 (and (= main.n main.n.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.3.next (and (not .s.2.next) (and (not .s.1.next) (not .s.0.next))))) (a!19 (and (and (not .s.3) .s.2 (and (not .s.1) (not .s.0))) (<= main.j main.n))) (a!21 (and (not (<= main.j main.n)) (and (not .s.3) (and (not .s.2) (and .s.1 .s.0))))) (a!22 (and (<= main.j main.n) (and (not .s.3) (and (not .s.2) (and .s.1 .s.0))))) (a!23 (and (not .s.3) (and (not .s.2) (and .s.1 (not .s.0))))) (a!25 (and (and (and (not .s.1) .s.0) (not .s.2)) (not .s.3))) (a!26 (and (= main.n main.n.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i.next 0) (not .s.3.next) (and (not .s.2.next) (and .s.1.next (not .s.0.next))))) (a!27 (and (not .s.3) (and (not .s.2) (and (not .s.1) (not .s.0)))))) (let ((a!8 (or (not (and a!7 (<= 0 main.i))) (and (= main.n main.n.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.3.next (not .s.2.next) (and .s.0.next (not .s.1.next))))) (a!9 (not (and a!7 (not (<= 0 main.i))))) (a!11 (or (not (and (not .s.3) .s.2 (and .s.1 .s.0))) a!10)) (a!15 (or (not a!14) (and (= main.n main.n.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.3.next) .s.2.next (and .s.1.next (not .s.0.next))))) (a!20 (or (not a!19) (and (= main.n main.n.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.3.next) .s.2.next (and .s.0.next (not .s.1.next))))) (a!24 (or (not a!23) (and (= main.n main.n.next) (= main.j main.j.next) (= main.i main.i.next) (= main.k.next 0) (and (and .s.0.next .s.1.next) (not .s.2.next)) (not .s.3.next))))) (let ((a!28 (and (state .s.1 .s.0 .s.2 .s.3 main.i main.j main.k main.n) (or a!1 a!2) (or (not a!3) a!4) (or (not a!5) a!6) a!8 (or a!2 a!9) a!11 (or a!12 (and (= main.n main.n.next) (= main.k main.k.next) (= main.i main.i.next) a!13 (not .s.3.next) (and .s.0.next .s.1.next) .s.2.next)) a!15 (or a!4 (not a!16)) (or (not a!17) a!18) a!20 (or a!4 (not a!21)) (or a!10 (not a!22)) a!24 (or (not a!25) a!26) (or a!4 (not a!27))))) (=> a!28 (state .s.1.next .s.0.next .s.2.next .s.3.next main.i.next main.j.next main.k.next main.n.next)))))))
(constraint (forall ((.s.1 Bool) (.s.0 Bool) (.s.2 Bool) (.s.3 Bool) (main.i Int) (main.j Int) (main.k Int) (main.n Int)) (let ((a!1 (not (not (and .s.3 (not .s.2) .s.1 .s.0))))) (=> (and (state .s.1 .s.0 .s.2 .s.3 main.i main.j main.k main.n) a!1) false))))

(check-synth)

