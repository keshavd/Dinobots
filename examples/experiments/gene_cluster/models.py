from dinobots.models.geometric_based.RevGNN import RevGNN


def get_model(num_features, num_classes, num_layers=10, num_hidden_channels=100):
    model = RevGNN(
        in_channels=num_features,
        hidden_channels=num_hidden_channels,
        out_channels=num_classes,
        num_layers=num_layers,  # You can try 1000 layers for fun
        dropout=0.5,
        num_groups=2,
    )
    return model
