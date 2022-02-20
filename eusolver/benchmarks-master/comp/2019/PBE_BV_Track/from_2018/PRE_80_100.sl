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

(constraint (= (f #x8359e4a9266b3000) #x74707d0c472e1cff))
(constraint (= (f #xbb18e613e8776e5e) #x39358b8ad9011abc))
(constraint (= (f #x2a3d9ce430ab7533) #xd31e894d8c49d379))
(constraint (= (f #xc010b701e947b60e) #x33ee3d8df823ce91))
(constraint (= (f #xee8d618ec8b4a2e5) #xee8d618ec8b4a2e4))
(constraint (= (f #xcb2a5dd18b0ec64e) #x2822fc515c404d4d))
(constraint (= (f #xc45cec2b0bea1e3e) #x2f5d451243573fde))
(constraint (= (f #x9b7c0c3114c953d0) #x9b7c0c3114c953d0))
(constraint (= (f #xac6851cada5bb63e) #x48d1291877fe8e5e))
(constraint (= (f #x809cc09c559a49b0) #x77597359e50c11b4))
(constraint (= (f #x1e57ce1ba8c42092) #x1e57ce1ba8c42092))
(constraint (= (f #x086acceede2e49b6) #xf70e864233eed1ae))
(constraint (= (f #xe4abe812ee39d71e) #xe4abe812ee39d71e))
(constraint (= (f #x776eb2400bb036bc) #x776eb2400bb036bc))
(constraint (= (f #x4c1a9a32d0833ece) #xaf23bc2a02748d45))
(constraint (= (f #xd6e09a6a08e9e8b6) #xd6e09a6a08e9e8b6))
(constraint (= (f #x52e04eb5ee2e5567) #xa7f1ac5eb2eec542))
(constraint (= (f #xda5ecc0e053b9116) #x17fb47311a70b5d8))
(constraint (= (f #x84b5bde47aeab462) #x72fee63d3d66a057))
(constraint (= (f #x5e730d07ab8e606e) #x9ba5c227d9b8b98b))
(constraint (= (f #xe048ce0650ea6a15) #x11b2a5194a06ef49))
(constraint (= (f #x2e4e77be470ee7a7) #xcecca0c5d48029de))
(constraint (= (f #x2693201964bee0e4) #xd703ade504f5310d))
(constraint (= (f #x60decce4b346e64a) #x9913464d0184ab51))
(constraint (= (f #x210dac0de502e553) #xdce179313cacec57))
(constraint (= (f #x68039cdc212658a3) #x917c29561cc741d2))
(constraint (= (f #x1a3139a1ea085c20) #x1a3139a1ea085c20))
(constraint (= (f #x9c6e4927aeab4ee8) #x59cad245d669fc29))
(constraint (= (f #x4e6672bed351cb72) #x4e6672bed351cb72))
(constraint (= (f #x80335dec08849240) #x80335dec08849240))
(constraint (= (f #x7cc912d79e18910c) #x7cc912d79e18910c))
(constraint (= (f #xed3577a01c4a6ee4) #x03f730e5e1f0ea2d))
(constraint (= (f #x05ce10d5e7d530ed) #x05ce10d5e7d530ec))
(constraint (= (f #xea72d668831d5d1c) #xea72d668831d5d1c))
(constraint (= (f #x0e45ee3744a4acd8) #x0e45ee3744a4acd8))
(constraint (= (f #xe4011c559129c09b) #xe4011c559129c098))
(constraint (= (f #xd44e982d8b57d954) #x1e6c7e4f9bf2a916))
(constraint (= (f #x08eed61561a94b69) #x08eed61561a94b68))
(constraint (= (f #x0c197e1da2b93268) #x0c197e1da2b93268))
(constraint (= (f #xbe45272c4a50d94e) #xbe45272c4a50d94e))
(constraint (= (f #x10184727549e2d16) #xeee634663617f018))
(constraint (= (f #xa4248e8b263b4e36) #x5199288c2760fce6))
(constraint (= (f #xd0c038814be30173) #x2233c3f69f5ece75))
(constraint (= (f #x0ecb3604e6a63541) #xf048169acaef676a))
(constraint (= (f #xee3119e3289d5324) #xee3119e3289d5324))
(constraint (= (f #x5cc20d5a9e39843c) #x5cc20d5a9e39843c))
(constraint (= (f #xe0b2dd3dcb39ae7d) #xe0b2dd3dcb39ae7c))
(constraint (= (f #x4937c01e0a8e0ac2) #xb234c3e014c91491))
(constraint (= (f #xe895b01e22e4e91e) #xe895b01e22e4e91e))
(constraint (= (f #x221002c7b59e2956) #xdbcefd0bcf07f414))
(constraint (= (f #xe4cd983da92e9d87) #x0ce58e3e7c3e78a0))
(constraint (= (f #x811e030268ee6692) #x76d01ccd7082b304))
(constraint (= (f #xe4ae16c4631d1244) #xe4ae16c4631d1244))
(constraint (= (f #x50c7e24121ab5448) #xaa2b9f9acc39f673))
(constraint (= (f #xbd5a64096dd8207d) #xbd5a64096dd8207c))
(constraint (= (f #x9350b79aee72acb4) #x637a3ceb62a62880))
(constraint (= (f #x229aed97c3ee019c) #xdb3b638ebfd31e4a))
(constraint (= (f #x15cae461d293ec40) #xe8d86d581042d4fb))
(constraint (= (f #x8be511e54386910e) #x6b5c9cfc684105e1))
(constraint (= (f #xee485e595e3dc7d5) #xee485e595e3dc7d4))
(constraint (= (f #xa2e267d633ec7123) #xa2e267d633ec7120))
(constraint (= (f #x7a3516ddee82ce27) #x7e2797b4329504f6))
(constraint (= (f #x3abc493852c36cc5) #xc197f23428105c6e))
(constraint (= (f #x6bcc11bd5a633e27) #x8d772d26cff68df6))
(constraint (= (f #x89496e0001a7c154) #x6e21fb1ffe3dc296))
(constraint (= (f #x318708ab30428313) #xcb6086ca1cb954bb))
(constraint (= (f #xca6db596420e2570) #x28eb6f1059d0f838))
(constraint (= (f #x74ec70d32e3e5122) #x83c4c81f9eddc9cb))
(constraint (= (f #xe5e8abaddb5a1cd0) #x0bb8c99746f04162))
(constraint (= (f #xcbadebe627734e10) #x2797355b76157d0e))
(constraint (= (f #x6e48e531bcb0e13c) #x6e48e531bcb0e13c))
(constraint (= (f #xce6119bee0b423e7) #xce6119bee0b423e0))
(constraint (= (f #xb207737ab777b5da) #x42d8154d9d10cec8))
(constraint (= (f #x8d85505196e18d85) #x8d85505196e18d84))
(constraint (= (f #x0d2d6de2bc3a5ec9) #xf1ffbb3f1801fb4a))
(constraint (= (f #xc40c87e479843977) #xc40c87e479843970))
(constraint (= (f #xca478c8dacaa1beb) #x2913faa9788b4256))
(constraint (= (f #x06b8cb901465e263) #x06b8cb901465e260))
(constraint (= (f #xe7437b285951862e) #xe7437b285951862e))
(constraint (= (f #xb16ea405d965dc04) #xb16ea405d965dc04))
(constraint (= (f #x2e89eb356d2a6484) #xce8d76173c02f533))
(constraint (= (f #xc031cc67639c87ad) #xc031cc67639c87ac))
(constraint (= (f #x51e839ed5837a4c1) #xa8f94273d244e0f2))
(constraint (= (f #x958c8482a8ee46b9) #x611ab3352c82d4db))
(constraint (= (f #x8e0b459107ed760d) #x8e0b459107ed760c))
(constraint (= (f #x913e0e628d88dda1) #x913e0e628d88dda0))
(constraint (= (f #x0bc05a2ee65d4e95) #x0bc05a2ee65d4e94))
(constraint (= (f #xe4e939a57a0e5b2d) #x0cc832c02e50bf20))
(constraint (= (f #x4d5766236932cb02) #xadd3237a603a084d))
(constraint (= (f #xdc8e1003e9ae3b0e) #x15a90efbd7b6e141))
(constraint (= (f #xe33c7477e27494e3) #xe33c7477e27494e0))
(constraint (= (f #x62ec1e6e14ade1ee) #x62ec1e6e14ade1ee))
(constraint (= (f #x60156b9109360124) #x99e93db5e6369ec9))
(constraint (= (f #x51c936c80eecac84) #x51c936c80eecac84))
(constraint (= (f #x78307c69050eee9a) #x804c7bd06aa0227c))
(constraint (= (f #x324b3788c695dd03) #x324b3788c695dd00))
(constraint (= (f #xe6ec8d902e215478) #xe6ec8d902e215478))
(constraint (= (f #x458a087427ec1e8a) #x458a087427ec1e8a))
(constraint (= (f #x2b02ab35439b9808) #xd24d2a17682aae77))
(constraint (= (f #xe565eb86d5543b36) #xe565eb86d5543b36))

(check-synth)

