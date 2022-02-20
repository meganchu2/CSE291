(set-logic BV)

(synth-fun max2 ((x (_ BitVec 32)) (y (_ BitVec 32))) (_ BitVec 32))

(declare-var x (_ BitVec 32))
(declare-var y (_ BitVec 32))
(constraint (bvuge (max2 x y) x))
(constraint (bvuge (max2 x y) y))
(constraint (or (= x (max2 x y)) (= y (max2 x y))))

(check-synth)

