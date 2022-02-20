
(set-logic BV)

(define-fun ehad ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000001))
(define-fun arba ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000004))
(define-fun shesh ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000010))
(define-fun smol ((x (BitVec 64))) (BitVec 64) (bvshl x #x0000000000000001))
(define-fun im ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64) (ite (= x #x0000000000000001) y z))

(synth-fun f ( (x (BitVec 64))) (BitVec 64)
(

(Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start)
                    (smol Start)
 		    (ehad Start)
		    (arba Start)
		    (shesh Start)
		    (bvand Start Start)
		    (bvor Start Start)
		    (bvxor Start Start)
		    (bvadd Start Start)
		    (im Start Start Start)
 ))
)
)


(constraint (= (f #x24a4b3e9377a04a6) #x000047804ce00008))
(constraint (= (f #xde7cce270138eb3b) #xde7cde7fcf3feb3b))
(constraint (= (f #xb1dabc927ee69e6e) #x43207000f9883898))
(constraint (= (f #xc9ee2365907cedcd) #x0398048200f19310))
(constraint (= (f #x11235436ba5e07c3) #x0004004860380f04))
(constraint (= (f #x34e9cd8ad7bc08e3) #x418312010e700184))
(constraint (= (f #x861cc41e6b69eb9c) #x861cc61eef7febfd))
(constraint (= (f #x1643ebb48831bee1) #x0807864000427980))
(constraint (= (f #xe3937ece0cd46e61) #x8604f91811009880))
(constraint (= (f #x09189d6ece4871ae) #x002030991800c218))
(constraint (= (f #xea3085799b8e80b7) #xea30ef799fff9bbf))
(constraint (= (f #xe87b25cc9db7e51e) #xe87bedffbdfffdbf))
(constraint (= (f #x9693c573a4cc4eed) #x080700c601101990))
(constraint (= (f #xbeee897e789ea7d2) #xbeeebffef9feffde))
(constraint (= (f #xd51ca4641beca5ba) #xd51cf57cbfecbffe))
(constraint (= (f #x0665ed046091b806) #x0883900080026008))
(constraint (= (f #x92dd44ee3ecd2c65) #x0130019879101080))
(constraint (= (f #x0a0eae832dee81ba) #x0a0eae8fafefadfe))
(constraint (= (f #xebba151c8e037a7d) #xebbaffbe9f1ffe7f))
(constraint (= (f #x23ee66454e0e411c) #x23ee67ef6e4f4f1e))
(constraint (= (f #xeb108ec40daeb284) #x8400190012184000))
(constraint (= (f #xd46cb4c475b3bc2d) #x00904100c2467010))
(constraint (= (f #x3de3472e29eab2c3) #x73840c1803804104))
(constraint (= (f #x4e0b12b054ea4b9a) #x4e0b5ebb56fa5ffa))
(constraint (= (f #x6120d3956807e48b) #x80010600800f8004))
(constraint (= (f #xde7d21e06c07ce06) #x38f00380900f1808))
(constraint (= (f #x5086139c6c28b9cb) #x0008063090006304))
(constraint (= (f #xeae1b8bb08724d8a) #x8182606400c01200))
(constraint (= (f #xe256bb626ec7402a) #x80086480990c0000))
(constraint (= (f #x67a18b90d5e37ccc) #x8e0206010384f110))
(constraint (= (f #xa902b9ead35da346) #x0000638104320408))
(constraint (= (f #x995bb7888ceaebc0) #x20264e0011818700))
(constraint (= (f #xdd6358a2ed621677) #xdd63dde3fde2ff77))
(constraint (= (f #x17c664e9ec2d5511) #x17c677efecedfd3d))
(constraint (= (f #xe4e0ed789daa586e) #x818190e032002098))
(constraint (= (f #x042cbd98a921e125) #x0010722000038000))
(constraint (= (f #x60e2e70c199ee557) #x60e2e7eeff9efddf))
(constraint (= (f #x37e697dee6d27e17) #x37e6b7fef7defed7))
(constraint (= (f #xa7ce38c0a2ebc873) #xa7cebfcebaebeafb))
(constraint (= (f #x0e9969898a641b2e) #x1820820200802418))
(constraint (= (f #xb9d341343900993e) #xb9d3f9f77934b93e))
(constraint (= (f #x4d2e3d16791a5952) #x4d2e7d3e7d1e795a))
(constraint (= (f #x7db05a8973932583) #xf2402000c6040204))
(constraint (= (f #xe14ec7e8c1e37c5e) #xe14ee7eec7ebfdff))
(constraint (= (f #xec14580e37e2e038) #xec14fc1e7feef7fa))
(constraint (= (f #xeed1dbc8e6e64e7b) #xeed1ffd9ffeeeeff))
(constraint (= (f #x6b2ee6026aa3ce19) #x6b2eef2eeea3eebb))
(constraint (= (f #x9cae4ee6ce4885ea) #x3018198918000380))
(constraint (= (f #x2946e94c983ee699) #x2946e94ef97efebf))
(constraint (= (f #x5a2462824175b269) #x2000800000c24080))
(constraint (= (f #x7e92cba5c7487884) #xf80106030c00e000))
(constraint (= (f #xe2ca2ccdbdece282) #x8100111273918000))
(constraint (= (f #x531632d73250a6ed) #x0408410c40000990))
(constraint (= (f #xa27e547ba8c46909) #x00f800e601008000))
(constraint (= (f #xae1e2e9d24640875) #xae1eae9f2efd2c75))
(constraint (= (f #x5407e545024992ca) #x000f800000020100))
(constraint (= (f #x3914c04ee6d098e7) #x600100198900218c))
(constraint (= (f #x87ee10e141cd6726) #x0f98018003108c08))
(constraint (= (f #x19a49c08a7715790) #x19a49dacbf79f7f1))
(constraint (= (f #xe4377700e7cccd52) #xe437f737f7ccefde))
(constraint (= (f #x960a92217de35912) #x960a962bffe37df3))
(constraint (= (f #x592eba985e462683) #x2018602038080804))
(constraint (= (f #x1481105e43465751) #x148114df535e5757))
(constraint (= (f #x04e9eceeed110b8a) #x0183919990000600))
(constraint (= (f #x7d43093ecaa977d2) #x7d437d7fcbbffffb))
(constraint (= (f #x788e82ed8d2ade9b) #x788efaef8fefdfbb))
(constraint (= (f #x9ad7957907c9806c) #x210e00e00f020090))
(constraint (= (f #x2c24e8179d0aa50d) #x1001800e30000010))
(constraint (= (f #x65be49226238c199) #x65be6dbe6b3ae3b9))
(constraint (= (f #xae269d0e533184e1) #x1808301804420180))
(constraint (= (f #xcdc733e061898164) #x130c478082020080))
(constraint (= (f #x505c3592d65a7e15) #x505c75def7dafe5f))
(constraint (= (f #x8ec13a381240e32a) #x1900606000018400))
(constraint (= (f #xcce2ee0b32a96d88) #x1181980440009200))
(constraint (= (f #x19ac098c53257b8c) #x221002100400e610))
(constraint (= (f #x21e08cee024666c2) #x0380119800088900))
(constraint (= (f #xe02e0232ead79cec) #x80180041810e3190))
(constraint (= (f #x0d301470da052ab8) #x0d301d70de75fabd))
(constraint (= (f #x29e05003ec60d777) #x29e079e3fc63ff77))
(constraint (= (f #x6e8536ce01c5a141) #x9800491803020000))
(constraint (= (f #x38dc4308e0ab8129) #x6130040180060000))
(constraint (= (f #xd52547642aa50e72) #xd525d7656fe52ef7))
(constraint (= (f #x52199423ae67e8a6) #x00220006188f8008))
(constraint (= (f #x803e710e6e84e05b) #x803ef13e7f8eeedf))
(constraint (= (f #x4ec90ec9544e62cc) #x1900190000188110))
(constraint (= (f #xce547e1ab304d368) #x1800f82044010480))
(constraint (= (f #x9a170194ac3db373) #x9a179b97adbdbf7f))
(constraint (= (f #x841eeb59902600ee) #x0039842200080198))
(constraint (= (f #x0ea5e873284c2bdb) #x0ea5eef7e87f2bdf))
(constraint (= (f #x1543009082260716) #x154315d382b68736))
(constraint (= (f #x44914ee04c3299d3) #x44914ef14ef2ddf3))
(constraint (= (f #x2ee13e4290e074e3) #x198078000180c184))
(constraint (= (f #x9ca9360142dd8c77) #x9ca9bea976ddceff))
(constraint (= (f #xc0a95eca7de6db4c) #x00003900f3892410))
(constraint (= (f #xede489ca4e66d524) #x9380030018890000))
(constraint (= (f #xed79494c75864c83) #x90e00010c2081004))
(constraint (= (f #x20d0cade83c2670d) #x0101013807008c10))
(constraint (= (f #xd189da708538461a) #xd189dbf9df78c73a))
(constraint (= (f #x08739bae3be016ec) #x00c6261867800990))
(constraint (= (f #xa1e7e338c64d3333) #xa1e7e3ffe77df77f))
(check-synth)
