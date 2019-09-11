import os
import tensorflow as tf

from NormNet import NormNet

def getconfig():
    config = {}
    config['device_id'] = '2'
    config['source_domain'] = {'name':'sfy-3d1',
                               'dataset_path':'../dataset/DA_dataset_dirlist_K/sfy-3d1_train_4wnplus.txt'}
    config['target_domain'] = {'name':'tongji-szsq',
                               'dataset_path':'../dataset/DA_dataset_dirlist_K/tongji-szsq_train.txt'}
    
    config['model'] = {}
    config['model']['generator'] = {'gf_dim':32,'lr_g':0.0001,
          'pretrained_path':'../tf_formal/ckpt/NormNet_tongji-szsqTOsfy-3d1_L1_100_d1_1.0_d2_1_task_0/NormNet-14'}
    config['model']['discriminator_intra'] = {'df_dim':32,'lr_d_intra':0.0001,'lambda_L_d_intra':1.,
          'pretrained_path':'../tf_formal/ckpt/NormNet_tongji-szsqTOsfy-3d1_L1_100_d1_1.0_d2_1_task_0/NormNet-14'}
    config['model']['discriminator_inter'] = {'df_dim':32,'lr_d_inter':0.0001,'lambda_L_d_inter':0.5,
          'pretrained_path':'../tf_formal/ckpt/NormNet_tongji-szsqTOsfy-3d1_L1_100_d1_1.0_d2_1_task_0/NormNet-14'}
    config['model']['tasknet'] = {'num_classes':1,'lr_tasknet':0.001,'lambda_L_task':1.,
          'pretrained_path':'../tf_formal/pretrained_models/tongji-szsq/tasknet6-0'}
    config['model']['lambda_L1'] = 100
    
    config['batchsize'] = 10
    config['buffer_size'] = 100
    config['start_epoch'] = 0
    config['epoch'] = 15
    config['optimizer'] = {'func':tf.train.AdamOptimizer,'parameters':{'beta1':0.3,'beta2':0.999}}
    config['ganloss'] = None
    
    config['input'] = {'size':512,'channel':4}
    config['output'] = {'size':512,'channel':3}
    
    name = 'NormNet_%sTO%s_L1_%s_d1_%s_d2_%s_task_%s'%(config['target_domain']['name'],
                         config['source_domain']['name'],
                         str(config['model']['lambda_L1']),
                         str(config['model']['discriminator_intra']['lambda_L_d_intra']),
                         str(config['model']['discriminator_inter']['lambda_L_d_inter']),
                         str(config['model']['tasknet']['lambda_L_task']))
    config['log_path'] = './logs/%s'%name
    config['ckpt_path'] = './ckpt/%s'%name
    
    return config


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = getconfig()
    os.environ['CUDA_VISIBLE_DEVICES'] = args['device_id']
    
    if not os.path.exists(args['log_path']):
        os.makedirs(args['log_path'])
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    
    with open(os.path.join(args['log_path'],'config.txt'),'w') as f:
        f.write(str(args))
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        model = NormNet(sess,args)
        model.train()
    
