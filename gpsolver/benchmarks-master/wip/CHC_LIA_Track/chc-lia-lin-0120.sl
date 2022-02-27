(set-logic CHC_LIA)

(synth-fun state ((x_0 Bool) (x_1 Bool) (x_2 Bool) (x_3 Bool) (x_4 Bool) (x_5 Bool) (x_6 Int) (x_7 Int) (x_8 Int) (x_9 Int) (x_10 Int) (x_11 Int)) Bool)

(constraint (forall ((.s.1 Bool) (.s.0 Bool) (.s.2 Bool) (.s.3 Bool) (.s.4 Bool) (.s.5 Bool) (main.i Int) (main.j Int) (main.k Int) (main.l Int) (main.m Int) (main.n Int)) (=> (and (not .s.1) .s.0 (not .s.2) (not .s.3) (not .s.4) (not .s.5)) (state .s.1 .s.0 .s.2 .s.3 .s.4 .s.5 main.i main.j main.k main.l main.m main.n))))
(constraint (forall ((.s.1 Bool) (.s.0 Bool) (.s.2 Bool) (.s.3 Bool) (.s.4 Bool) (.s.5 Bool) (main.i Int) (main.j Int) (main.k Int) (main.l Int) (main.m Int) (main.n Int) (.s.0.next Bool) (.s.1.next Bool) (.s.2.next Bool) (.s.3.next Bool) (.s.4.next Bool) (.s.5.next Bool) (main.i.next Int) (main.j.next Int) (main.k.next Int) (main.l.next Int) (main.m.next Int) (main.n.next Int) (.inputVar.0 Int)) (let ((a!1 (and (not .s.4) (and .s.3 (and (not .s.2) (and .s.1 .s.0))))) (a!2 (and (and (and (and .s.0.next .s.1.next) (not .s.2.next)) .s.3.next) (not .s.4.next))) (a!4 (and .s.3 (and (not .s.2) (and .s.1 (not .s.0))))) (a!6 (and (not .s.3.next) (and (not .s.2.next) (and (not .s.1.next) (not .s.0.next))))) (a!8 (and (and (and (not .s.1) .s.0) (not .s.2)) .s.3)) (a!10 (and .s.3.next (and (not .s.2.next) (and .s.1.next (not .s.0.next))))) (a!12 (and .s.3 (and (not .s.2) (and (not .s.1) (not .s.0))))) (a!13 (+ main.j (* (- 1) main.k) (* (- 1) main.n))) (a!15 (and .s.3.next (and (not .s.2.next) (and .s.0.next (not .s.1.next))))) (a!18 (and (not .s.4) (and (not .s.3) (and .s.2 (and .s.1 .s.0))))) (a!19 (and .s.3.next (and .s.2.next (and .s.1.next (not .s.0.next))))) (a!21 (and (not .s.3) (and .s.2 (and .s.1 (not .s.0))))) (a!23 (and (not .s.4.next) (and (not .s.3.next) (and (and .s.0.next .s.1.next) .s.2.next)))) (a!25 (and (not .s.3) (and (and (not .s.1) .s.0) .s.2))) (a!27 (and (not .s.3.next) (and (not .s.2.next) (and .s.1.next (not .s.0.next))))) (a!29 (and (not .s.3) (and .s.2 (and (not .s.1) (not .s.0))))) (a!31 (and (not .s.3.next) (and .s.2.next (and .s.0.next (not .s.1.next))))) (a!32 (= (+ main.l (* (- 1) main.l.next)) (- 1))) (a!34 (and (not .s.4) (and (not .s.3) (and (not .s.2) (and .s.1 .s.0))))) (a!35 (= (+ main.j (* (- 1) main.j.next)) 1)) (a!36 (and (not .s.3.next) (and .s.2.next (and (not .s.1.next) (not .s.0.next))))) (a!38 (and (not .s.3) (and (not .s.2) (and .s.1 (not .s.0))))) (a!40 (and (not .s.3.next) (and .s.2.next (and .s.1.next (not .s.0.next))))) (a!43 (and (not .s.4.next) (and (and (and .s.0.next .s.1.next) (not .s.2.next)) (not .s.3.next)))) (a!44 (and (and (and (not .s.1) .s.0) (not .s.2)) (not .s.3))) (a!47 (and (not .s.3) (and (not .s.2) (and (not .s.1) (not .s.0))))) (a!49 (and (not .s.3.next) (and (not .s.2.next) (and .s.0.next (not .s.1.next))))) (a!51 (and (not .s.5) .s.4 (and .s.3 (and .s.2 (and .s.1 .s.0))))) (a!53 (and .s.3 (and .s.2 (and .s.1 (not .s.0))))) (a!54 (= (+ main.i (* (- 1) main.i.next)) (- 1))) (a!57 (and .s.3 (and (and (not .s.1) .s.0) .s.2))) (a!58 (and (not .s.5.next) .s.4.next (and (not .s.3.next) (and (and .s.0.next .s.1.next) .s.2.next)))) (a!60 (and .s.3 (and .s.2 (and (not .s.1) (not .s.0))))) (a!61 (and .s.3.next (and .s.2.next (and .s.0.next (not .s.1.next))))) (a!63 (and (not .s.5) .s.4 (and .s.3 (and (not .s.2) (and .s.1 .s.0))))) (a!65 (= (+ main.j (* (- 1) main.j.next)) (- 1))) (a!70 (and .s.3.next (and .s.2.next (and (not .s.1.next) (not .s.0.next))))) (a!72 (and (not .s.5) .s.4 (and (not .s.3) (and .s.2 (and .s.1 .s.0))))) (a!74 (and .s.3.next (and (not .s.2.next) (and (not .s.1.next) (not .s.0.next))))) (a!77 (and (not .s.5.next) .s.4.next (and (and (and .s.0.next .s.1.next) (not .s.2.next)) (not .s.3.next)))) (a!80 (and (not .s.5) .s.4 (and (not .s.3) (and (not .s.2) (and .s.1 .s.0))))) (a!89 (and (not .s.5) (not .s.4) (and .s.3 (and .s.2 (and .s.1 .s.0))))) (a!90 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) (not .s.4.next) (and .s.3.next (and (and .s.0.next .s.1.next) .s.2.next))))) (let ((a!3 (and a!2 .s.5.next (= main.i main.i.next) (= main.j main.j.next) (= main.k main.k.next) (= main.l main.l.next) (= main.m main.m.next) (= main.n main.n.next))) (a!5 (not (and .s.5 (and (not .s.4) a!4)))) (a!7 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) (and (not .s.4.next) a!6))) (a!9 (not (and .s.5 (and (not .s.4) a!8)))) (a!14 (and (and .s.5 (and (not .s.4) a!12)) (<= a!13 0))) (a!17 (and (and .s.5 (and (not .s.4) a!12)) (not (<= a!13 0)))) (a!20 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) .s.4.next a!19)) (a!22 (not (and .s.5 (and (not .s.4) a!21)))) (a!24 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next a!23)) (a!26 (not (and .s.5 (and (not .s.4) a!25)))) (a!28 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (and .s.5.next (and (not .s.4.next) a!27)))) (a!30 (not (and .s.5 (and (not .s.4) a!29)))) (a!37 (or (not (and .s.5 a!34)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.i main.i.next) a!35 .s.5.next (and (not .s.4.next) a!36)))) (a!39 (and (and .s.5 (and (not .s.4) a!38)) (<= main.n main.l))) (a!42 (and (and .s.5 (and (not .s.4) a!38)) (not (<= main.n main.l)))) (a!45 (not (and (and a!44 (not .s.4)) .s.5))) (a!46 (and (= main.n main.n.next) (= main.m main.m.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (and .s.5.next (and (not .s.4.next) a!27)) (= main.l.next 0))) (a!48 (and (and .s.5 (and (not .s.4) a!47)) (not (<= main.k main.n)))) (a!50 (and (and .s.5 (and (not .s.4) a!47)) (<= main.k main.n))) (a!52 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (and (not .s.5.next) (and (not .s.4.next) a!36)))) (a!55 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (not .s.5.next) .s.4.next (and .s.3.next (and (and .s.0.next .s.1.next) .s.2.next)) a!54)) (a!59 (or (not (and (not .s.5) .s.4 a!57)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) a!58))) (a!62 (or (not (and (not .s.5) .s.4 a!60)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) a!32 (not .s.5.next) .s.4.next a!61))) (a!64 (or (not a!63) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (and (not .s.5.next) .s.4.next a!15)))) (a!66 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.i main.i.next) (not .s.5.next) (and (and (and .s.0.next .s.1.next) (not .s.2.next)) .s.3.next) .s.4.next a!65)) (a!68 (not (and (and (not .s.5) .s.4 a!8) (not (<= 0 a!13))))) (a!69 (not (and (and (not .s.5) .s.4 a!8) (<= 0 a!13)))) (a!71 (or (not (and (not .s.5) .s.4 a!12)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.i main.i.next) (and (not .s.5.next) .s.4.next a!15) (= main.j.next 0)))) (a!73 (not (and a!72 (not (<= main.m main.l))))) (a!76 (or (not (and (not .s.5) .s.4 a!21)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (= main.l.next 0) a!58))) (a!78 (or (not (and (not .s.5) .s.4 a!25)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) a!77))) (a!79 (or (not (and (not .s.5) .s.4 a!29)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) a!32 (not .s.5.next) .s.4.next a!31))) (a!81 (not (and (not (<= main.m main.l)) a!80))) (a!82 (or (not (and (<= main.m main.l) a!80)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) .s.4.next a!40))) (a!83 (or (not (and (not .s.5) .s.4 a!38)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (= main.l.next 0) a!77))) (a!84 (not (and (and (not .s.5) a!44 .s.4) (not (<= main.k 5))))) (a!85 (not (and (and (not .s.5) a!44 .s.4) (<= main.k 5)))) (a!87 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (and (not .s.5.next) (and (not .s.4.next) a!40)))) (a!91 (or (not (and (not .s.5) (not .s.4) a!53)) a!90)) (a!92 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (and (not .s.5.next) (and (not .s.4.next) a!10)))) (a!94 (or (not (and (not .s.5) (not .s.4) a!60)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) a!32 (not .s.5.next) (not .s.4.next) a!61))) (a!95 (or (not (and (not .s.5) a!1)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.i main.i.next) a!65 (not .s.5.next) (not .s.4.next) a!70))) (a!96 (and (<= main.n main.l) (and (not .s.5) (and (not .s.4) a!4)))) (a!97 (and (not (<= main.n main.l)) (and (not .s.5) (and (not .s.4) a!4)))) (a!98 (not (and (not .s.5) (and (not .s.4) a!8)))) (a!99 (and (= main.n main.n.next) (= main.m main.m.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (= main.l.next 0) (and (not .s.5.next) (and (not .s.4.next) a!10)))) (a!100 (and (and (not .s.5) (and (not .s.4) a!12)) (= .inputVar.0 0))) (a!102 (and (and (not .s.5) (and (not .s.4) a!12)) (not (= .inputVar.0 0)))) (a!103 (or (not (and (not .s.5) a!18)) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.i main.i.next) (= main.j.next main.k) (not .s.5.next) (and (not .s.4.next) a!74)))) (a!104 (and (and (not .s.5) (and (not .s.4) a!21)) (not (<= main.m main.i)))) (a!105 (and (and (not .s.5) (and (not .s.4) a!21)) (<= main.m main.i))) (a!106 (not (and (not .s.5) (and (not .s.4) a!25)))) (a!107 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (and (not .s.5.next) (and (not .s.4.next) a!40)) (= main.i.next 0))) (a!108 (and (and (not .s.5) (and (not .s.4) a!29)) (not (<= main.n main.i)))) (a!110 (and (and (not .s.5) (and (not .s.4) a!29)) (<= main.n main.i))) (a!112 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (and (not .s.5.next) (and (not .s.4.next) a!36)) (= main.i.next 0))) (a!114 (not (and (not .s.5) (and (not .s.4) a!38)))) (a!115 (and (and (and a!44 (not .s.4)) (not .s.5)) (<= a!13 0))) (a!116 (and (and (and a!44 (not .s.4)) (not .s.5)) (not (<= a!13 0)))) (a!118 (not (and (not .s.5) (and (not .s.4) a!47))))) (let ((a!11 (or a!9 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next (and (not .s.4.next) a!10)))) (a!16 (or (not a!14) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next (and (not .s.4.next) a!15)))) (a!33 (or a!30 (and (= main.n main.n.next) (= main.m main.m.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next (and (not .s.4.next) a!31) a!32))) (a!41 (or (not a!39) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next (and (not .s.4.next) a!40)))) (a!56 (or (not (and (not .s.5) .s.4 a!53)) a!55)) (a!67 (or (not (and (not .s.5) .s.4 a!4)) a!66)) (a!75 (or a!20 (not (and a!72 (<= main.m main.l))))) (a!86 (or a!85 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next (and (not .s.4.next) a!6)))) (a!88 (or (not (and (not .s.5) .s.4 a!47)) a!87)) (a!93 (or (not (and (not .s.5) (not .s.4) a!57)) a!92)) (a!101 (or (not a!100) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) (and (not .s.4.next) a!15)))) (a!109 (or (not a!108) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) (and (not .s.4.next) a!31)))) (a!111 (or (not a!110) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next (and (not .s.4.next) a!74)))) (a!113 (or (not (and (not .s.5) a!34)) a!112)) (a!117 (or (not a!116) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) (and (not .s.4.next) a!27))))) (let ((a!119 (and (state .s.1 .s.0 .s.2 .s.3 .s.4 .s.5 main.i main.j main.k main.l main.m main.n) (or (not (and .s.5 a!1)) a!3) (or a!5 a!7) a!11 a!16 (or a!3 (not a!17)) (or (not (and .s.5 a!18)) a!20) (or a!22 a!24) (or a!26 a!28) a!33 a!37 a!41 (or (not a!42) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next a!43)) (or a!45 a!46) (or (not a!48) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) .s.5.next (not .s.4.next) a!49)) (or a!24 (not a!50)) (or (not a!51) a!52) a!56 a!59 a!62 a!64 a!67 (or a!68 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) .s.4.next a!10)) (or a!69 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) .s.4.next a!70)) a!71 (or a!73 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) .s.4.next a!74)) a!75 a!76 a!78 a!79 (or a!81 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) .s.4.next a!36)) a!82 a!83 (or a!84 (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) .s.4.next a!27)) a!86 a!88 (or (not a!89) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) a!54 (not .s.5.next) .s.4.next a!6)) a!91 a!93 a!94 a!95 (or (not a!96) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) (not .s.4.next) a!19)) (or (not a!97) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) a!2 (not .s.5.next))) (or a!98 a!99) a!101 (or a!90 (not a!102)) a!103 (or (not a!104) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) a!23)) (or (not a!105) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) .s.4.next a!49)) (or a!106 a!107) a!109 a!111 a!113 (or a!7 a!114) (or (not a!115) (and (= main.n main.n.next) (= main.m main.m.next) (= main.l main.l.next) (= main.k main.k.next) (= main.j main.j.next) (= main.i main.i.next) (not .s.5.next) a!43)) a!117 (or a!7 a!118)))) (=> a!119 (state .s.1.next .s.0.next .s.2.next .s.3.next .s.4.next .s.5.next main.i.next main.j.next main.k.next main.l.next main.m.next main.n.next))))))))
(constraint (forall ((.s.1 Bool) (.s.0 Bool) (.s.2 Bool) (.s.3 Bool) (.s.4 Bool) (.s.5 Bool) (main.i Int) (main.j Int) (main.k Int) (main.l Int) (main.m Int) (main.n Int)) (let ((a!1 (not (not (and .s.5 (not .s.4) .s.3 (not .s.2) .s.1 .s.0))))) (=> (and (state .s.1 .s.0 .s.2 .s.3 .s.4 .s.5 main.i main.j main.k main.l main.m main.n) a!1) false))))

(check-synth)

