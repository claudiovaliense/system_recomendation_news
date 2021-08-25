#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Sistema de recomendacao para noticias no dominio de tecnologia
import claudio_funcoes as cv
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn.metrics.pairwise as cos  # Calculate similarity cosine
import torch
import numpy as np
import statistics
import sys
import scipy.stats as stats # Calcular intervalo de confiança


def arredonda(number, precisao=2):
    """ Arredonda number in precision. Example: arredonda(2.1234, 2); Return='2.12'"""
    return float(f'%.{precisao}f'%(number))

def ic(tamanho, std, confianca, type='t', lado=2):
    """Calcula o intervalo de confianca"""
    if lado is 1:
        lado = (1 - confianca) # um lado o intervalo fica mais estreito
    else:
        lado = (1 - confianca) /2 
        
    #print(f'Valor de t: {stats.t.ppf(1- (lado), tamanho-1) }')    
    if type is 'normal':
        return stats.norm.ppf(1 - (lado)) * ( std / ( tamanho ** (1/2) ) )
    return stats.t.ppf(1- (lado), tamanho-1) * ( std / ( tamanho ** (1/2) ) ) 

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

def mean_reciprocal_rank_claudio(rs):        
    rr = []
    for r in rs:
        try: 
            r = r.index(1) +1
        except:
            r = 130
            #r = sys.maxsize # considerar que o elemento nao existe
            #r = len(r)+1 # considerar que o item relevante esta na proxima posicao depois do k        
        rr.append(1 / r )
    return arredonda(statistics.mean(rr) ), arredonda(ic(len(rr), statistics.stdev(rr), 0.95) )
    #return arredonda(100* statistics.mean(rr) ), arredonda(100* ic(len(rr), statistics.stdev(rr), 0.95) )


def mean_reciprocal_rank(rs):    
    rs = (np.asarray(r).nonzero()[0] for r in rs)       
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
    

def precision_at_k(r, k):    
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):    
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):    
    return np.mean([average_precision(r) for r in rs])

def read_data():
    articles = cv.arquivo_para_corpus_delimiter(f'dataset/archive/shared_articles.csv', ',')#, 1000) # 0 timestamp, 2 contenId, 3 authorPersonId,  10 title, 11 text
    user = cv.arquivo_para_corpus_delimiter(f'dataset/archive/users_interactions.csv', ',')#, 1000)     

    # indices: 0 timestamp, 2 identificacao conteudo, 3 identificacao autor, 10 titulo, -1 lang
    articles_dict = dict()   # dicionario onde as chaves sao as identificacoes dos autores 
    for index_row in range(1, len( articles)):
        if articles[index_row][-1] != 'en': # filter english
            continue
        if articles_dict.get( articles[index_row][3] ) == None: 
            articles_dict[articles[index_row][3]] = []
        articles_dict[articles[index_row][3]].append( {'timestamp': datetime.fromtimestamp( int( articles[index_row][0])) , 'contenId' : articles[index_row][2], 'title' : articles[index_row][10]}) #, 'text' : articles[index_row][11] })

    chave_2 = []
    for k, v in articles_dict.items():
        if len(v)>=2:
            chave_2.append( k)
    
    print(len(chave_2))
    
    
    x_train = []; x_test = []; ground_truth = []
    for k in chave_2:
        qtd_noticias_user = len(articles_dict[k])
        #if qtd_noticias_user > 5:
        #    qtd_noticias_user= 5
        #print(qtd_noticias_user)
        #qtd_noticias_user = 2
        titles = []
        limit_k = 2; cont=0
        for index_noticia_user in range(qtd_noticias_user-1): # -1 porque a ultima é teste        
            if cont == limit_k: 
                break
            cont+=1
            #titles.append( representation_bert( [ articles_dict[k][index_noticia_user]['title'] ], 'bert_concat') [0] )
            #titles.append( articles_dict[k][index_noticia_user]['title'] )            
            titles.append( articles_dict[k][qtd_noticias_user-2 - index_noticia_user]['title'] ) # pega as noticias mais recentes com excessao da ultima
        
        #temp = np.mean( representation_bert( titles, 'bert_avg') , axis=0)
        #print(temp.shape); exit()
        #x_train.append( np.mean( representation_bert( titles, 'bert_avg') , axis=0) )
        #x_train.append( np.mean(titles, axis=0) )
        #print(x_train); exit()
        x_train.append( " ".join( titles ))
        #x_train.append( " ".join( [articles_dict[k][0]['title'], articles_dict[k][1]['title'] ] ))
        x_test.append( articles_dict[k][-1]['title'])
        #x_test.append( representation_bert( [ articles_dict[k][-1]['title'] ], 'bert_avg')[0]  )
    
    #for i in range( len( x_train)): print(f'{i}: {x_train[i]}')
    #for i in range( len( x_test)): print(f'{i}: {x_test[i]}')
    
    x_train = [cv.preprocessor(x) for x in x_train]
    x_test = [cv.preprocessor(x) for x in x_test]
    #x_train = representation_bert(x_train, 'bert_concat')
    #x_test = representation_bert(x_test, 'bert_concat')

    #np.save('temp/x_train_all_bert_ultimas', x_train); np.save('temp/x_test_all_bert_ultimas', x_test)
    x_train = np.load('temp/x_train_all_bert_ultimas.npy');x_test = np.load('temp/x_test_all_bert_ultimas.npy')
    #x_train = np.load('temp/x_train3.npy'); x_test = np.load('temp/x_test3.npy')
    #x_train = np.load('temp/x_train_all.npy'); x_test = np.load('temp/x_test_all.npy')
    

    #for i in range( len( x_test)): print(f'{i}: {cv.arredonda(100*(cos.cosine_similarity( [x_train[0]], [x_test[i]] ))[0][0])} ')
    predicoes_all = []
    k = 130# limitar a olhar as k posicoes
    for avaliar in range(len(x_train)):
        escore = []
        #avaliar = 10 # avaliar usuario com indice 10, o indice 10 é onde esta o elemento relevante para o usuario
        for i in range( len( x_test)): 
            escore.append( cv.arredonda(100*(cos.cosine_similarity( [x_train[avaliar]], [x_test[i]] ))[0][0]) )
        topk = cv.k_max_index2(escore, k)
        
        predicao = []
        valido = False; index_certo = -1
        for index_top in range( len(topk)):
            if topk[index_top] in [avaliar]:
                valido = True
                index_certo= index_top
                predicao.append(1)
            else:
                predicao.append(0)
        #if valido == True:            
        predicoes_all.append(predicao)
        print(avaliar, topk, index_certo)
    print(len(predicoes_all))
    print(f'MRR: {mean_reciprocal_rank_claudio(predicoes_all)[0]} +/- {mean_reciprocal_rank_claudio(predicoes_all)[1]}')
    print(f'MRR: {mean_reciprocal_rank(predicoes_all)} ')
    #print(f'MAP: {mean_average_precision(predicoes_all)}')
    #print(f'Escore top: {escore[topk[0]]}, escore gabarito: {escore[avaliar]}')    


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

