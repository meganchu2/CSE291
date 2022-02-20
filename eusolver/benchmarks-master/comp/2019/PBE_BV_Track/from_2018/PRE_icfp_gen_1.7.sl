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

(constraint (= (f #x7236FD323FD10280) #x8DC902CDC02EFD7F))
(constraint (= (f #xD6F0505231B5DBAC) #x290FAFADCE4A2453))
(constraint (= (f #xEFC9E80E697B5CEC) #x103617F19684A313))
(constraint (= (f #xACB5DA70A8B675E0) #x534A258F57498A1F))
(constraint (= (f #x6228E36147FCEFC0) #x9DD71C9EB803103F))
(constraint (= (f #x26179EC36430620D) #x4C2F3D86C860C41A))
(constraint (= (f #xF96B331B25BAE619) #xF2D666364B75CC32))
(constraint (= (f #x71BE3C1E5175B951) #xE37C783CA2EB72A2))
(constraint (= (f #x35093BB72738130D) #x6A12776E4E70261A))
(constraint (= (f #x5ECC1CED799D9B49) #xBD9839DAF33B3692))
(constraint (= (f #xABE3B09C5EFF712E) #x541C4F63A1008ED1))
(constraint (= (f #xC0DF2757FE7190DA) #x3F20D8A8018E6F25))
(constraint (= (f #x12230347B45E66BE) #xEDDCFCB84BA19941))
(constraint (= (f #xADB6DCA00DF63CA2) #x5249235FF209C35D))
(constraint (= (f #x8847539637922D56) #x77B8AC69C86DD2A9))
(constraint (= (f #x1AF162B4158D9784) #xE50E9D4BEA72687B))
(constraint (= (f #xC2167B47246AF2E8) #x3DE984B8DB950D17))
(constraint (= (f #x2E62602A606B8EE4) #xD19D9FD59F94711B))
(constraint (= (f #xC1483865348B1D5C) #x3EB7C79ACB74E2A3))
(constraint (= (f #x2FF464B5FF497A44) #xD00B9B4A00B685BB))
(constraint (= (f #x14DC7B66B41EEDBF) #x29B8F6CD683DDB7E))
(constraint (= (f #x4935F10EBE9E20FB) #x926BE21D7D3C41F6))
(constraint (= (f #x4818379C649108FB) #x90306F38C92211F6))
(constraint (= (f #xD592AAA6C07BFC23) #xAB25554D80F7F846))
(constraint (= (f #x5E270177CF18693B) #xBC4E02EF9E30D276))
(constraint (= (f #x1F75967C9FCEB381) #x1F75967C9FCEB381))
(constraint (= (f #x44ECBD1794069489) #x44ECBD1794069489))
(constraint (= (f #x59C6F3B0A48EEE25) #x59C6F3B0A48EEE25))
(constraint (= (f #xAF8E26B062824A59) #xAF8E26B062824A59))
(constraint (= (f #x29F865225A67DE65) #x29F865225A67DE65))
(constraint (= (f #xAC0ADE5785018366) #x53F521A87AFE7C99))
(constraint (= (f #x20A240F7F98F6172) #xDF5DBF0806709E8D))
(constraint (= (f #x4D49875F08CC3292) #xB2B678A0F733CD6D))
(constraint (= (f #x5DD78883C347BC8E) #xA228777C3CB84371))
(constraint (= (f #xB026CDEBA2A7BBE2) #x4FD932145D58441D))
(constraint (= (f #x388551C28AE0245B) #x388551C28AE0245B))
(constraint (= (f #xBC42CFBFF7627D6B) #xBC42CFBFF7627D6B))
(constraint (= (f #xD67C53C673634E27) #xD67C53C673634E27))
(constraint (= (f #xEBF97B64ED641803) #xEBF97B64ED641803))
(constraint (= (f #x017430768E67D24F) #x017430768E67D24F))
(constraint (= (f #xFFFFFFFFFFFFFFFF) #xFFFFFFFFFFFFFFFE))
(constraint (= (f #x1BE88589BA201842) #xE4177A7645DFE7BD))
(constraint (= (f #x49EA2AE53E599623) #x93D455CA7CB32C46))
(constraint (= (f #xEA82CC5E6104247D) #xEA82CC5E6104247D))
(constraint (= (f #x75820D31BED79B87) #xEB041A637DAF370E))
(constraint (= (f #xE682665199EE31A8) #x197D99AE6611CE57))
(constraint (= (f #x9D8D9C6595EE5DED) #x9D8D9C6595EE5DED))
(constraint (= (f #xAD1B863E6B5351D4) #x52E479C194ACAE2B))
(constraint (= (f #xA7465C5C466DE212) #x58B9A3A3B9921DED))
(constraint (= (f #xC287ECB0E2E8EB85) #xC287ECB0E2E8EB85))
(constraint (= (f #xAC30404490729C8C) #x53CFBFBB6F8D6373))
(constraint (= (f #x51EAD8D97C522039) #xA3D5B1B2F8A44072))
(constraint (= (f #xDD949F185B6BD961) #xDD949F185B6BD961))
(constraint (= (f #xECA17E099D1D5B5D) #xD942FC133A3AB6BA))
(constraint (= (f #x073C03534C092263) #x073C03534C092263))
(constraint (= (f #xFFFFFFFFFFFFFFFF) #xFFFFFFFFFFFFFFFE))
(constraint (= (f #x9AA7E45F6673B2AE) #x65581BA0998C4D51))

(check-synth)

