class ConfigMDIC:

    global_path = "src_MDIC"
    blip_model = "Salesforce/blip2-opt-2.7b-coco"
    # blip_model = "Salesforce/blip2-opt-2.7b"
    max_number_tokens = 32
    

    # MDIC
    target_rate = 0.2031
    target_rate_cor = 0.2031
    rate_cfg_cor = {}  

    rate_cfg_cor[0.2031] = (64, 8196) 
    rate_cfg_cor[0.1875] = (64, 4096)
    rate_cfg_cor[0.1719] = (64, 2048)
    rate_cfg_cor[0.1563] = (64, 1024)
    rate_cfg_cor[0.1406] = (64, 512)    
    rate_cfg_cor[0.1250] = (64, 256)
    rate_cfg_cor[0.0937] = (64, 64)
    rate_cfg_cor[0.0507] = (32, 8196)
    rate_cfg_cor[0.0313] = (32, 256)
    rate_cfg_cor[0.0098] = (16, 1024)
    rate_cfg_cor[0.0024] = (8, 1024)
    rate_cfg_cor[0.0019] = (8, 256)
    rate_cfg = {}  

    rate_cfg[0.2031] = (64, 8196)
    rate_cfg[0.1875] = (64, 4096)
    rate_cfg[0.1719] = (64, 2048)
    rate_cfg[0.1563] = (64, 1024)
    rate_cfg[0.1406] = (64, 512)  
    rate_cfg[0.1250] = (64, 256)
    rate_cfg[0.0937] = (64, 64)
    rate_cfg[0.0507] = (32, 8196)#
    rate_cfg[0.0313] = (32, 256)
    rate_cfg[0.0098] = (16, 1024)
    rate_cfg[0.0024] = (8, 1024)
    rate_cfg[0.0019] = (8, 256)
    # {v_prediction, epsilon}
    prediction_type = "v_prediction"
    lpips_weight = 0.1
    guidance_scale = 3.0
    # probability of dropping text-conditioning
    cond_drop_prob = 0.1
    # number of sampling steps
    num_inference_steps = 10  

    random_seed = 3868512668962463
    
    # lambd = 0.1
    lambd = 10
