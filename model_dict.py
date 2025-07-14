from models import ProPINN


def get_model(args):
    model_dict = {
        'ProPINN': ProPINN,
    }
    return model_dict[args.model]
