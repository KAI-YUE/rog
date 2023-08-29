import argparse

parser = argparse.ArgumentParser(description='Class Selective Loss for Partial Multi Label Classification.')

# parser.add_argument('--model_path', type=str, default='./models_local/mtresnet_opim_86.72.pth')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dataset_type', type=str, default='OpenImages')
parser.add_argument('--class_description_path', type=str, default='data/oidv6-class-descriptions.csv')
parser.add_argument('--th', type=float, default=0.8)
parser.add_argument('--top_k', type=float, default=60)

def parse_args(parser=parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args