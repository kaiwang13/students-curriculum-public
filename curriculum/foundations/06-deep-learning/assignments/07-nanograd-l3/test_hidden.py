from solution import MLP, Value, train_xor


def test_mlp_forward_shape():
    model = MLP(2, [4, 1], seed=0)
    out = model([0.5, -0.5])
    assert isinstance(out, Value)


def test_params_count():
    model = MLP(2, [4, 1], seed=0)   # layer1: 4*(2+1)=12, layer2: 1*(4+1)=5 -> 17
    assert len(model.parameters()) == 17


def test_xor_learns():
    model, final_loss = train_xor(steps=200, lr=0.1, seed=1)
    assert final_loss < 0.5          # a from-scratch MLP solves XOR
    for x, y in [([0.0, 0.0], -1.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], -1.0)]:
        pred = model(x).data
        assert (pred > 0) == (y > 0)  # correct sign on all four
