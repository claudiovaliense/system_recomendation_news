#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Sistema de recomendacao para noticias no dominio de tecnologia
import claudio_funcoes as cv
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn.metrics.pairwise as cos  # Calculate similarity cosine
import torch

def assign_GPU(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda:0')    
    attention_mask = Tokenizer_output['attention_mask'].to('cuda:0')

    output = {'input_ids' : tokens_tensor,
          #'token_type_ids' : token_type_ids,
          'attention_mask' : attention_mask}

    return output

def representation_bert(x, pooling=None):
    """Create representation BERT"""
    import numpy
    from transformers import BertModel, BertTokenizer
    
    if "16" in pooling: limit_token=16
    elif "32" in pooling: limit_token=32
    elif "64" in pooling: limit_token=64
    elif "128" in pooling: limit_token=128
    elif "256" in pooling: limit_token=256
    elif "512" in pooling: limit_token=512
    limit_token=512
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model = model.to('cuda:0') # gpu    
    for index_doc in range(len(x)):  

        inputs = tokenizer(x[index_doc], return_tensors="pt", max_length=limit_token, truncation=True) 
        inputs = assign_GPU(inputs)        
        outputs = model(**inputs)
        
        if 'bert_concat' in pooling or 'bert_sum' in pooling or 'bert_last_avg' in pooling or 'bert_cls' in pooling:
            hidden_states = outputs[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1) # remove a primeira dimensao que do embedding incial
            token_embeddings = token_embeddings.permute(1,0,2) # reordena para em cada linha ser um token diferente
            vets = []
            for token in token_embeddings:
                if 'bert_concat' == pooling:
                    vets.append( torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0).cpu().detach().numpy() ) # concatena as 4 ultimas dimensoes
                elif 'bert_sum' == pooling:
                    vets.append( torch.sum(token[-4:], dim=0).cpu().detach().numpy() )
                elif 'bert_last_avg' == pooling:
                    vets.append( torch.mean(token[-4:], dim=0).cpu().detach().numpy() )
                elif 'bert_cls' == pooling:
                    x[index_doc] = token[-1].cpu().detach().numpy() # o primeiro  token é o cls e ultima camada
                    break
            
            if 'bert_cls' != pooling:
                x[index_doc] = numpy.mean( vets, axis=0)
            
        else:
            tokens = outputs[0].cpu().detach().numpy()[0]            
            if 'bert_avg' in pooling:
                x[index_doc] = numpy.mean(tokens, axis=0) #average
            elif 'bert_max' in pooling: x[index_doc] = numpy.amax(tokens, axis=0)    
    return x

def print_box_plot(data, interval_y=None):
    """Show boxplot"""
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.boxplot(data )
    ax.set_ylim(interval_y)
    #plt.show()
    plt.savefig('fig/fig.png', format='png') 
#print_box_plot([1, 5, 25, 54, 3], [0,30])

def read_data():
    articles = cv.arquivo_para_corpus_delimiter(f'dataset/archive/shared_articles.csv', ',')#, 1000) # 0 timestamp, 2 contenId, 3 authorPersonId,  10 title, 11 text
    user = cv.arquivo_para_corpus_delimiter(f'dataset/archive/users_interactions.csv', ',')#, 1000)     

    articles_dict = dict()    
    for index_row in range(1, len( articles)):
        if articles[index_row][-1] != 'en': # filter english
            continue
        if articles_dict.get( articles[index_row][3] ) == None: 
            articles_dict[articles[index_row][3]] = []
        articles_dict[articles[index_row][3]].append( {'timestamp': datetime.fromtimestamp( int( articles[index_row][0])) , 'contenId' : articles[index_row][2], 'title' : articles[index_row][10]}) #, 'text' : articles[index_row][11] })

    chave_2 = []
    for k, v in articles_dict.items():
        if len(v)==2:
            chave_2.append( k)
    
    print(len(chave_2))
    
    
    x_train = []; x_test = []; ground_truth = []
    for k in chave_2:
        x_train.append( articles_dict[k][0]['title'])
        x_test.append( articles_dict[k][1]['title'])
    
    for i in range( len( x_train)): print(f'{i}: {x_train[i]}')
    for i in range( len( x_test)): print(f'{i}: {x_test[i]}')
    
    x_train = [cv.preprocessor(x) for x in x_train]
    x_test = [cv.preprocessor(x) for x in x_test]
    x_train = representation_bert(x_train, 'bert_concat')
    x_test = representation_bert(x_test, 'bert_concat')

    #for i in range( len( x_test)): print(f'{i}: {cv.arredonda(100*(cos.cosine_similarity( [x_train[0]], [x_test[i]] ))[0][0])} ')
    escore = []
    avaliar = 10
    for i in range( len( x_test)): 
        escore.append( cv.arredonda(100*(cos.cosine_similarity( [x_train[avaliar]], [x_test[i]] ))[0][0]) )
    topk = cv.k_max_index2(escore, 20)
    print(topk)
    print(f'Escore top: {escore[topk[0]]}, escore gabarito: {escore[avaliar]}')    


if __name__ == "__main__":
    read_data()

# --- backup
'''
publicaram = []
    for k, v in articles_dict.items():
        publicaram.append(len(v))
    print_box_plot(publicaram, [0, 30]); exit()

    publicacoes = []
    for k, v in articles_dict.items():
        for i in range(len(v)):
            publicacoes.append( v[i]['timestamp'].month )
    print_box_plot(publicacoes); exit()


'''

