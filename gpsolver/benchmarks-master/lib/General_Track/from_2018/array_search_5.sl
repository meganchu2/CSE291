(set-logic LIA)

(synth-fun findIdx ((y1 Int) (y2 Int) (y3 Int) (y4 Int) (y5 Int) (k1 Int)) Int
    ((Start Int) (BoolExpr Bool))
    ((Start Int (0 1 2 3 4 5 y1 y2 y3 y4 y5 k1 (ite BoolExpr Start Start)))
    (BoolExpr Bool ((< Start Start) (<= Start Start) (> Start Start) (>= Start Start)))))

(declare-var x1 Int)
(declare-var x2 Int)
(declare-var x3 Int)
(declare-var x4 Int)
(declare-var x5 Int)
(declare-var k Int)
(constraint (=> (and (< x1 x2) (and (< x2 x3) (and (< x3 x4) (< x4 x5)))) (=> (< k x1) (= (findIdx x1 x2 x3 x4 x5 k) 0))))
(constraint (=> (and (< x1 x2) (and (< x2 x3) (and (< x3 x4) (< x4 x5)))) (=> (> k x5) (= (findIdx x1 x2 x3 x4 x5 k) 5))))
(constraint (=> (and (< x1 x2) (and (< x2 x3) (and (< x3 x4) (< x4 x5)))) (=> (and (> k x1) (< k x2)) (= (findIdx x1 x2 x3 x4 x5 k) 1))))
(constraint (=> (and (< x1 x2) (and (< x2 x3) (and (< x3 x4) (< x4 x5)))) (=> (and (> k x2) (< k x3)) (= (findIdx x1 x2 x3 x4 x5 k) 2))))
(constraint (=> (and (< x1 x2) (and (< x2 x3) (and (< x3 x4) (< x4 x5)))) (=> (and (> k x3) (< k x4)) (= (findIdx x1 x2 x3 x4 x5 k) 3))))
(constraint (=> (and (< x1 x2) (and (< x2 x3) (and (< x3 x4) (< x4 x5)))) (=> (and (> k x4) (< k x5)) (= (findIdx x1 x2 x3 x4 x5 k) 4))))

(check-synth)

