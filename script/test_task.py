# coding: utf-8


def test_start(test_task, log_path):
    test_task.mode()
    pairs = test_task.pairs
    correct = 0

    for inp, tar in pairs:
        out = test_task.model(inp)
        _, topi = out.topk(1)
        pd = topi.item()

        if pd == tar:
            correct += 1

    accuracy = correct/len(pairs)
    print("Accuracy: " + str(accuracy))

    with open(log_path + ".txt", 'w', encoding='utf-8') as f:
        f.write("Accuracy: " + str(accuracy) + "\n")
