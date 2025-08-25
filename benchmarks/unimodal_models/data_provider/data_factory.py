from benchmarks.unimodal_models.data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    if flag == 'test':
        drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        root_path=args.root_path, flag=flag,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        features=args.features, data_path=args.data_path,
        target=args.target, timeenc=timeenc, freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
