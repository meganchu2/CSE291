(set-logic BV)

(synth-fun f_141-11-141-28 ((x.end (BitVec 32)) (size (BitVec 32)) (i (BitVec 32)) (elem.end (BitVec 32))) Bool)

(declare-var x.end_141-11-141-28 (BitVec 32))
(declare-var size_141-11-141-28 (BitVec 32))
(declare-var i_141-11-141-28 (BitVec 32))
(declare-var elem.end_141-11-141-28 (BitVec 32))
(constraint (=> (and (= size_141-11-141-28 #x00000003) (and (= x.end_141-11-141-28 #x0000007a) (and (= elem.end_141-11-141-28 #x0000006e) (= i_141-11-141-28 #x00000002)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) true)))
(constraint (or (=> (and (= size_141-11-141-28 #x00000002) (and (= x.end_141-11-141-28 #x0000000f) (and (= elem.end_141-11-141-28 #x0000000f) (= i_141-11-141-28 #x00000001)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) false)) (=> (and (= size_141-11-141-28 #x00000002) (and (= x.end_141-11-141-28 #x0000000f) (and (= elem.end_141-11-141-28 #x0000000f) (= i_141-11-141-28 #x00000001)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) true))))
(constraint (=> (and (= size_141-11-141-28 #x00000003) (and (= x.end_141-11-141-28 #x0000007a) (and (= elem.end_141-11-141-28 #x0000006e) (= i_141-11-141-28 #x00000002)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) true)))
(constraint (=> (and (= size_141-11-141-28 #x00000003) (and (= x.end_141-11-141-28 #x0000007a) (and (= elem.end_141-11-141-28 #x0000006f) (= i_141-11-141-28 #x00000002)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) true)))
(constraint (=> (and (= size_141-11-141-28 #x00000002) (and (= x.end_141-11-141-28 #x00000003) (and (= elem.end_141-11-141-28 #x00000004) (= i_141-11-141-28 #x00000001)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) false)))
(constraint (=> (and (= size_141-11-141-28 #x00000002) (and (= x.end_141-11-141-28 #x0000007a) (and (= elem.end_141-11-141-28 #x0000006d) (= i_141-11-141-28 #x00000001)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) true)))
(constraint (=> (and (= size_141-11-141-28 #x00000002) (and (= x.end_141-11-141-28 #x0000012c) (and (= elem.end_141-11-141-28 #x00000131) (= i_141-11-141-28 #x00000001)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) false)))
(constraint (=> (and (= size_141-11-141-28 #x00000002) (and (= x.end_141-11-141-28 #x00000002) (and (= elem.end_141-11-141-28 #x00000015) (= i_141-11-141-28 #x00000001)))) (= (f_141-11-141-28 x.end_141-11-141-28 size_141-11-141-28 i_141-11-141-28 elem.end_141-11-141-28) false)))

(check-synth)

