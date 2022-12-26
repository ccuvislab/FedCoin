def avgWeight(model_list,ratio_list):
    parties = len(model_list)
    model_tmp=[None] * parties
    #optims_tmp=[None] * parties

    for idx, my_model in enumerate(model_list):
        print(next(my_model.parameters()).device)
        
        model_tmp[idx] = my_model.state_dict()


    for key in model_tmp[0]:    
        #print(key)
        model_avg = 0

        for idx, model_tmp_content in enumerate(model_tmp):     # add each model              
            model_avg += ratio_list[idx] * model_tmp_content[key]
            
        for i in range(len(model_tmp)):  #copy to each model            
            model_tmp[i][key] = model_avg
    for i in range(len(model_list)):    
        model_list[i].load_state_dict(model_tmp[i])
        
    return model_list  #, optims_tmp
    
    
    
def avg_model_weight(model_list):
    balance = 1/len(model_list)
    ratio_list= [balance] * len(model_list)
    
    avg_model=avgWeight(model_list,ratio_list)
    return avg_model
