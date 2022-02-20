(set-logic LIA)

(synth-fun findIdx ((y1 Int) (y2 Int) (k1 Int)) Int
    ((Start Int (0 1 2 y1 y2 k1 (ite BoolExpr Start Start)))
    (BoolExpr Bool ((< Start Start) (<= Start Start) (> Start Start) (>= Start Start)))))

(declare-var x1 Int)
(declare-var x2 Int)
(declare-var k Int)
(constraint (=> (< x1 x2) (=> (< k x1) (= (findIdx x1 x2 k) 0))))
(constraint (=> (< x1 x2) (=> (> k x2) (= (findIdx x1 x2 k) 2))))
(constraint (=> (< x1 x2) (=> (and (> k x1) (< k x2)) (= (findIdx x1 x2 k) 1))))

(check-synth)

