data_dict = {
    1: 'Scene-15',
    2: 'LandUse-21',
    3: 'handwritten',
    4: 'MSRC_v1',
    5: 'ORL_mtv',
    6: 'NoisyMNIST',
}


def get_config(flag=1):
    """确定网络的参数信息"""
    data_name = data_dict[flag]
    if data_name in ['Scene-15']:
        return dict(
            dataset=data_name,
            n=4485,
            topk=30,
            bottom_k=30 * 5,
            diffusion_epoch= 5,
            n_clusters=15,
            views=3,
            view=[1, 2],
            training=dict(
                lr=1.0e-3,
                epoch=500,
            ),
            Autoencoder=dict(
                gcnEncoder1=[0, 1024, 1024, 1024, 1024 // 4],
                gcnEncoder2=[0, 1024, 1024, 1024, 1024 // 4],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            )
        )
    elif data_name in ['LandUse-21']:
        return dict(
            dataset=data_name,
            n=2100,
            topk=40,
            bottom_k=40 * 5,
            diffusion_epoch= 50,
            n_clusters=21,
            views=3,
            view=[1, 2],
            training=dict(
                lr=1.0e-3,
                epoch=500,
            ),
            Autoencoder=dict(
                gcnEncoder1=[0, 1024, 1024, 1024, 1024 // 16],
                gcnEncoder2=[0, 1024, 1024, 1024, 1024 // 16],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['handwritten']:
        return dict(
            dataset=data_name,
            n=2000,
            topk=75,
            bottom_k=75*5,
            diffusion_epoch= 50,
            n_clusters=10,
            views=6,
            view=[1, 4],
            training=dict(
                lr=1.0e-3,
                epoch=500
            ),
            Autoencoder=dict(
                gcnEncoder1=[0, 1024, 1024, 1024, 1024 // 32],
                gcnEncoder2=[0, 1024, 1024, 1024, 1024 // 32],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['MSRC_v1']:
        return dict(
            dataset=data_name,
            n=210,
            topk=15,
            bottom_k=15 * 5,
            diffusion_epoch= 50,
            n_clusters=7,
            views=5,
            view=[1, 2],
            training=dict(
                lr=1.0e-3,
                epoch=500
            ),
            Autoencoder=dict(
                gcnEncoder1=[0, 1024, 1024, 1024, 1024 // 1],
                gcnEncoder2=[0, 1024, 1024, 1024, 1024 // 1],
                activations1='relu',
                activations2='relu',
                batchnorm=True
            ),
        )
    elif data_name in ['ORL_mtv']:
        return dict(
            dataset=data_name,
            n=400,
            topk=10,
            bottom_k=10 * 5,
            diffusion_epoch= 50,
            n_clusters=40,
            views=3,
            view=[0, 1],
            training=dict(
                lr=1.0e-3,
                epoch=500
            ),
            Autoencoder=dict(
                gcnEncoder1=[0, 1024, 1024, 1024, 1024 // 4],
                gcnEncoder2=[0, 1024, 1024, 1024, 1024 // 4],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['NoisyMNIST']:
        return dict(
            dataset=data_name,
            n=10000,
            topk=7,
            bottom_k=10,
            diffusion_epoch= 10,
            n_clusters=10,
            views=2,
            view=[0, 1],
            training=dict(
                lr=1.0e-3,
                epoch=500
            ),
            Autoencoder=dict(
                gcnEncoder1=[0, 1024, 1024, 1024, 1024 // 1],
                gcnEncoder2=[0, 1024, 1024, 1024, 1024 // 1],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    else:
        raise Exception('Undefined data_name in Config')