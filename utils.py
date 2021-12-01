import numpy as np
import paddle


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None):
    
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader


def convert_example(example, tokenizer, is_test=False):

    input_ids = tokenizer.encode(example["text"])
    input_ids = np.array(input_ids, dtype='int64')

    if not is_test:
        label = np.array(example["label"], dtype="int64")
        return input_ids, label
    else:
        return input_ids


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):

    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()