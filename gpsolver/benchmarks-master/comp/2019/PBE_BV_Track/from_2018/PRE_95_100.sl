(set-logic BV)

(define-fun ehad ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (BitVec 64))) (BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (BitVec 64))) (BitVec 64)
    ((Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #xc63d828943639c3a) #x0000000000000000))
(constraint (= (f #x099533940e5c961d) #xf66acc6bf1a369e2))
(constraint (= (f #x9eae4851beaa3bed) #x6151b7ae4155c412))
(constraint (= (f #x3d92e46b94584c6a) #x3d92e46b94584c6a))
(constraint (= (f #xb9a60908b5c6716c) #x0000000000000000))
(constraint (= (f #xe6330114b821a66a) #x0000000000000000))
(constraint (= (f #xbe83054057355e98) #x0000000000000000))
(constraint (= (f #x01942646c262b70e) #x01942646c262b70e))
(constraint (= (f #xc43c7b27328d1d74) #xc43c7b27328d1d74))
(constraint (= (f #xe5ed0b69e6e14c7e) #x0000000000000000))
(constraint (= (f #x93bd0e9497e4c528) #x0000000000000000))
(constraint (= (f #x73569d07463c374e) #x73569d07463c374e))
(constraint (= (f #xabc6aae30b63bb06) #xabc6aae30b63bb06))
(constraint (= (f #x5c06dee2a808a691) #xa3f9211d57f7596e))
(constraint (= (f #x0364dae7ebeec46b) #xfc9b251814113b94))
(constraint (= (f #x05e0644233899d06) #x05e0644233899d06))
(constraint (= (f #x33209d88c6abaeeb) #xccdf627739545114))
(constraint (= (f #x7716eb157dd58e29) #x88e914ea822a71d6))
(constraint (= (f #xacd9c45ec0dda6ee) #xacd9c45ec0dda6ee))
(constraint (= (f #xe2148e4dada57870) #x0000000000000000))
(constraint (= (f #x2e7ab4ee331ba9a4) #x2e7ab4ee331ba9a4))
(constraint (= (f #xbd1078c780387060) #xbd1078c780387060))
(constraint (= (f #x4c4e5acc490ea6b2) #x0000000000000000))
(constraint (= (f #x1be265eb99ad6391) #xe41d9a1466529c6e))
(constraint (= (f #xe5dc3e9eeeeea1b1) #x1a23c16111115e4e))
(constraint (= (f #x81b3167ea4c575ed) #x7e4ce9815b3a8a12))
(constraint (= (f #x814a9a7d04dd7ee5) #x7eb56582fb22811a))
(constraint (= (f #x558c85c37b9dce56) #x558c85c37b9dce56))
(constraint (= (f #x7681636258b74e7a) #x7681636258b74e7a))
(constraint (= (f #x85d68e43356e4bb1) #x7a2971bcca91b44e))
(constraint (= (f #xe4e306e4becc6a97) #x1b1cf91b41339568))
(constraint (= (f #x4eb2bd057d37210b) #xb14d42fa82c8def4))
(constraint (= (f #x44dd5b3a92c6e59e) #x44dd5b3a92c6e59e))
(constraint (= (f #x03966ec22eee054e) #x03966ec22eee054e))
(constraint (= (f #xd17d39080076b97c) #x0000000000000000))
(constraint (= (f #x4e3dea06a8736e6e) #x4e3dea06a8736e6e))
(constraint (= (f #xd76e59586cbbc859) #x2891a6a7934437a6))
(constraint (= (f #xda929a4da0e40456) #x0000000000000000))
(constraint (= (f #x9795a8e2e71a77de) #x9795a8e2e71a77de))
(constraint (= (f #x1ce27b4d95b6522c) #x0000000000000000))
(constraint (= (f #x9977ba478e5865ec) #x9977ba478e5865ec))
(constraint (= (f #x44b59a9c1ce1d419) #xbb4a6563e31e2be6))
(constraint (= (f #x057a7872e6bc4a55) #xfa85878d1943b5aa))
(constraint (= (f #x5d91bae468e65365) #xa26e451b9719ac9a))
(constraint (= (f #x15ebec6cd4c846e1) #xea1413932b37b91e))
(constraint (= (f #x244de2d55512de6e) #x0000000000000000))
(constraint (= (f #x4466c2215eb383e1) #xbb993ddea14c7c1e))
(constraint (= (f #x902e6d1967a0d312) #x0000000000000000))
(constraint (= (f #x3392e4cbea6c7870) #x3392e4cbea6c7870))
(constraint (= (f #x5acc18ba47eb0d8c) #x5acc18ba47eb0d8c))
(constraint (= (f #x1808e378465955db) #xe7f71c87b9a6aa24))
(constraint (= (f #xe8b13e688844eb85) #x174ec19777bb147a))
(constraint (= (f #xe268ad583e25ca1e) #x0000000000000000))
(constraint (= (f #xe3a2e18c911c5e28) #x0000000000000000))
(constraint (= (f #xa8eb15b98be592e8) #x0000000000000000))
(constraint (= (f #xa38401364e94992e) #xa38401364e94992e))
(constraint (= (f #xdd47ed254ccd260e) #x0000000000000000))
(constraint (= (f #x46ec52acd8e7eae0) #x0000000000000000))
(constraint (= (f #x304708868912a265) #xcfb8f77976ed5d9a))
(constraint (= (f #xdd256e8ecd9dda9e) #xdd256e8ecd9dda9e))
(constraint (= (f #x386e1ebe80b3bd44) #x386e1ebe80b3bd44))
(constraint (= (f #xcee4b696ae5e86d8) #xcee4b696ae5e86d8))
(constraint (= (f #xcdb6867650deac66) #xcdb6867650deac66))
(constraint (= (f #xbae114553e50c8e4) #x0000000000000000))
(constraint (= (f #xe4cba1e7d6e0aa08) #xe4cba1e7d6e0aa08))
(constraint (= (f #x7e99426a1ce6d596) #x7e99426a1ce6d596))
(constraint (= (f #xd82a8bdaad2b21b1) #x27d5742552d4de4e))
(constraint (= (f #x64630d1a64a9eeae) #x64630d1a64a9eeae))
(constraint (= (f #x872b8a8e7ad968ae) #x872b8a8e7ad968ae))
(constraint (= (f #x0be2c253324804b2) #x0be2c253324804b2))
(constraint (= (f #xe89d7709e658e10e) #x0000000000000000))
(constraint (= (f #xb136e9a4e4484eb3) #x4ec9165b1bb7b14c))
(constraint (= (f #x0de7ae30dd279e11) #xf21851cf22d861ee))
(constraint (= (f #x521e41988a6470c2) #x0000000000000000))
(constraint (= (f #xbc43d819812d1a1a) #x0000000000000000))
(constraint (= (f #x016ecc64d18953a4) #x0000000000000000))
(constraint (= (f #xca549eba16e49120) #xca549eba16e49120))
(constraint (= (f #x4ce5a8b5dd36b931) #xb31a574a22c946ce))
(constraint (= (f #x2875205e3b1995d1) #xd78adfa1c4e66a2e))
(constraint (= (f #x7b4245ad6080a7cd) #x84bdba529f7f5832))
(constraint (= (f #x70c94e5cb679104b) #x8f36b1a34986efb4))
(constraint (= (f #xcedb460b30acead6) #xcedb460b30acead6))
(constraint (= (f #x845690a514216eec) #x0000000000000000))
(constraint (= (f #xddadceed523820dd) #x22523112adc7df22))
(constraint (= (f #xa0077e3b4e36cce2) #xa0077e3b4e36cce2))
(constraint (= (f #x7835ad9226489745) #x87ca526dd9b768ba))
(constraint (= (f #x75e241ae676e0e2a) #x75e241ae676e0e2a))
(constraint (= (f #xc8d69567c5eaeeee) #xc8d69567c5eaeeee))
(constraint (= (f #x90ab1c46e8ae6b0d) #x6f54e3b9175194f2))
(constraint (= (f #x80783581eab26767) #x7f87ca7e154d9898))
(constraint (= (f #x78217d205cee7878) #x0000000000000000))
(constraint (= (f #x7e7c9e6400aeddb4) #x0000000000000000))
(constraint (= (f #xddae811e2dd24091) #x22517ee1d22dbf6e))
(constraint (= (f #xb940eee4edca20a2) #x0000000000000000))
(constraint (= (f #x9b665713e816421a) #x9b665713e816421a))
(constraint (= (f #x210e93d7ecd9ec14) #x210e93d7ecd9ec14))
(constraint (= (f #x0b9c84d37336579d) #xf4637b2c8cc9a862))
(constraint (= (f #xe78c1d42b87469d7) #x1873e2bd478b9628))
(constraint (= (f #xe2b116e6d2e9e21e) #xe2b116e6d2e9e21e))
(constraint (= (f #xb3511245e929d3da) #x0000000000000000))

(check-synth)

