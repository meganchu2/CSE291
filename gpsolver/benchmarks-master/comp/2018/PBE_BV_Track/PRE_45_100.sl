
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


(constraint (= (f #xc5c227a5911ae4da) #x0000000000000000))
(constraint (= (f #x3d7a081b5ace667e) #x0000000000000000))
(constraint (= (f #xe2d85204b5e8a08d) #x00001d27adfb4a17))
(constraint (= (f #x5a078bab35514e71) #x0000a5f87454caae))
(constraint (= (f #xe4e6655c89b7c795) #x00001b199aa37648))
(constraint (= (f #x8176b253871522be) #x0000000000000000))
(constraint (= (f #xe56e2d5b1298b00e) #x0000000000000000))
(constraint (= (f #xec48cca90d5653a4) #x0000000000000000))
(constraint (= (f #x9c6b3eac3deecc91) #x00006394c153c211))
(constraint (= (f #x2497dd96e2800673) #x0000db6822691d7f))
(constraint (= (f #xa955bd2ae6eb3793) #x000056aa42d51914))
(constraint (= (f #xd172dea14ebc5613) #x00002e8d215eb143))
(constraint (= (f #x0136925ce8ad2e46) #x0000000000000000))
(constraint (= (f #x19e183d1832e6b55) #x0000e61e7c2e7cd1))
(constraint (= (f #x741b71949a436280) #x0000000000000000))
(constraint (= (f #x0900375e7a1bd338) #x0000000000000000))
(constraint (= (f #xb4723b12d6de2491) #x00004b8dc4ed2921))
(constraint (= (f #x756825e8c6098ad7) #x00008a97da1739f6))
(constraint (= (f #x5eb3eda408615332) #x0000000000000000))
(constraint (= (f #x2bb62ed861eed24d) #x0000d449d1279e11))
(constraint (= (f #x4243908cc86983eb) #x0000bdbc6f733796))
(constraint (= (f #x8d8c8c86a113b01e) #x0000000000000000))
(constraint (= (f #x222e4e8051dc3734) #x0000000000000000))
(constraint (= (f #xc85e8ea6e4503a22) #x0000000000000000))
(constraint (= (f #x6b52169910be0150) #x0000000000000000))
(constraint (= (f #x20a234de67de5119) #x0000df5dcb219821))
(constraint (= (f #xc164b855b9b615ab) #x00003e9b47aa4649))
(constraint (= (f #xe0aed3575e885199) #x00001f512ca8a177))
(constraint (= (f #x19b4c608771dc0c9) #x0000e64b39f788e2))
(constraint (= (f #xe9948b13e2ee0bdc) #x0000000000000000))
(constraint (= (f #xa55cee8eed2a6c57) #x00005aa3117112d5))
(constraint (= (f #x0aeebc2a4ed2cb77) #x0000f51143d5b12d))
(constraint (= (f #x4e87e5ba78e9acb5) #x0000b1781a458716))
(constraint (= (f #xd27e20737b0ae256) #x0000000000000000))
(constraint (= (f #xeee7e8c86a4d0084) #x0000000000000000))
(constraint (= (f #x67e98bd64e8818ee) #x0000000000000000))
(constraint (= (f #xb98a3b34e939b8ea) #x0000000000000000))
(constraint (= (f #xe092db7529ac7445) #x00001f6d248ad653))
(constraint (= (f #xcc8a702d27dca023) #x000033758fd2d823))
(constraint (= (f #x9240e53d47ae5137) #x00006dbf1ac2b851))
(constraint (= (f #x416cdadcc31a1070) #x0000000000000000))
(constraint (= (f #xc0ae880a384e92e6) #x0000000000000000))
(constraint (= (f #x5ca023e5de23823c) #x0000000000000000))
(constraint (= (f #xce6065a2d344708e) #x0000000000000000))
(constraint (= (f #x36b1c0b16eb6e8a0) #x0000000000000000))
(constraint (= (f #x15ebbc014b5772ca) #x0000000000000000))
(constraint (= (f #x6e533654bad2ab77) #x000091acc9ab452d))
(constraint (= (f #x90ee161297bd1e49) #x00006f11e9ed6842))
(constraint (= (f #x4e055ea6e6d0dec6) #x0000000000000000))
(constraint (= (f #xc673dae0e20a0ba0) #x0000000000000000))
(constraint (= (f #xbcb7630da856057a) #x0000000000000000))
(constraint (= (f #x47699da0c631ceee) #x0000000000000000))
(constraint (= (f #x4723d6ebc5e8b65d) #x0000b8dc29143a17))
(constraint (= (f #xce4071e355d109ee) #x0000000000000000))
(constraint (= (f #x64759337a0a68417) #x00009b8a6cc85f59))
(constraint (= (f #x4aee5b8895431a8a) #x0000000000000000))
(constraint (= (f #xdbb730090c22b065) #x00002448cff6f3dd))
(constraint (= (f #x16a3da7580d79608) #x0000000000000000))
(constraint (= (f #x2aaa0e0d1de6a419) #x0000d555f1f2e219))
(constraint (= (f #x319c8271747210eb) #x0000ce637d8e8b8d))
(constraint (= (f #xa5deb584d7639029) #x00005a214a7b289c))
(constraint (= (f #x1d1be83e2641d0dc) #x0000000000000000))
(constraint (= (f #xe2d0dbe757e9995b) #x00001d2f2418a816))
(constraint (= (f #xbe38e84bd8ce42d8) #x0000000000000000))
(constraint (= (f #x278b6138e5279745) #x0000d8749ec71ad8))
(constraint (= (f #x8b1e88e9937b698d) #x000074e177166c84))
(constraint (= (f #x9470817b576988eb) #x00006b8f7e84a896))
(constraint (= (f #x83524eb8e84e36a5) #x00007cadb14717b1))
(constraint (= (f #xc6aa9ede41e78015) #x000039556121be18))
(constraint (= (f #x8e0eddb54b2b2999) #x000071f1224ab4d4))
(constraint (= (f #xea315e6dd7930b63) #x000015cea192286c))
(constraint (= (f #xb227db1ed4581144) #x0000000000000000))
(constraint (= (f #x9d41b8a8ac587ed5) #x000062be475753a7))
(constraint (= (f #x82897b168eede9bd) #x00007d7684e97112))
(constraint (= (f #xe385c461a77ac0c1) #x00001c7a3b9e5885))
(constraint (= (f #xe04421a00813d63e) #x0000000000000000))
(constraint (= (f #x8cbcc516a34978a9) #x000073433ae95cb6))
(constraint (= (f #x35aeacae0d62de04) #x0000000000000000))
(constraint (= (f #xe5e7c3eca76de6a8) #x0000000000000000))
(constraint (= (f #xb4e7432b012408de) #x0000000000000000))
(constraint (= (f #x4666a61ceb170595) #x0000b99959e314e8))
(constraint (= (f #x82e0c3e04cede522) #x0000000000000000))
(constraint (= (f #xb26b18ee42e411c6) #x0000000000000000))
(constraint (= (f #xb19a8ae8c37e49e7) #x00004e6575173c81))
(constraint (= (f #x3e817e9c4c6debe7) #x0000c17e8163b392))
(constraint (= (f #x6b3e2dd8b98e7be1) #x000094c1d2274671))
(constraint (= (f #x61b266a0e9e3228b) #x00009e4d995f161c))
(constraint (= (f #x90588e5770e147eb) #x00006fa771a88f1e))
(constraint (= (f #x599e60d0a0ec4223) #x0000a6619f2f5f13))
(constraint (= (f #x3e618ea78cc8a489) #x0000c19e71587337))
(constraint (= (f #x56451c949e0836a4) #x0000000000000000))
(constraint (= (f #x60563b444293cb24) #x0000000000000000))
(constraint (= (f #xae5eaac975cb951a) #x0000000000000000))
(constraint (= (f #xd4ec9a2a4b028219) #x00002b1365d5b4fd))
(constraint (= (f #x56a980766192de8b) #x0000a9567f899e6d))
(constraint (= (f #x2250ec00ae06c149) #x0000ddaf13ff51f9))
(constraint (= (f #xb1e8ed4edec7e913) #x00004e1712b12138))
(constraint (= (f #xc7943de793eae558) #x0000000000000000))
(constraint (= (f #x450acb5472a13687) #x0000baf534ab8d5e))
(constraint (= (f #x9c11e468c2ddc873) #x000063ee1b973d22))
(check-synth)
