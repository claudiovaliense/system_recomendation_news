#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Sistema de recomendação
import sys
import numpy
import math
import statistics
import json
import unicodedata

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def cosine(x, y):
    EPSILON = 1e-07
    dot_products = numpy.dot(x, y.T)
    norm_products = numpy.linalg.norm(x) * numpy.linalg.norm(y)
    return dot_products / (norm_products + EPSILON)

def leitura(content, rating, target):    
    """Leitura dos dados."""
    rating = open(rating); content = open(content); target = open(target)
    next(rating); next(content); next(target) # jump head
    user_item = dict(); item_content = dict(); item_user = dict(); 
    #target_user_item = dict()
    target_user_item = []

    for line in target:
        line = line.strip().split(":")
        #target_user_item[line[0]] = line[1]
        target_user_item.append( (line[0] , line[1]) )

    cont=0#; notas = []
    for line in rating: # criar dicionario que tem as relacoes dos usuario, itens e notas
        if cont == 1000000: break
        cont+=1
        line = line.strip().split(",")
        user_item_line = line[0].split(":")
        #notas.append( int(line[1])) 
        
        #if item_content.get( user_item_line[1] ) == None: # so considera itens com conteudo
        #    continue
        #print(user_item_line[1])
        
        if user_item.get(user_item_line[0]) == None: 
            user_item[user_item_line[0]] = []
        user_item[user_item_line[0]].append( (user_item_line[1], int(line[1]) )) 
        
        if item_user.get(user_item_line[1]) == None:
            item_user[user_item_line[1]] = []
        item_user[user_item_line[1]].append( (user_item_line[0], int(line[1]) ) )
    
    #print(item_user)
    #print(item_content)
    

    cont = 0
    for line in content:
        if cont == 10000000: break
        cont+=1
        line = line.split(",",1)        
        if item_user.get(line[0]) == None:
            continue
        #line = line.split(",",1)
        #conteudo = json.loads( " ".join(line[1:]) )        
        #exit()
        #item_content[line[0]] =   line[1].split('"')[3].lower() # title 
        temp_json = json.loads("".join(line[1:]).lower())   
        item_content[line[0]] = temp_json 
        #print(temp_json) 
        
        #if temp_json.get('imdbrating') != None:
        #    item_content[line[0]] = temp_json['imdbrating']

    #print(len(user_item)); print(len(item_user.keys())); print(len(item_content.keys())); exit()
    return user_item, item_user, item_content, target_user_item

def matrix(user_item, item_users):
    """Cria a matrix user (linha) x itens (coluna)"""
    list_itens = []
    #matriz_user_item = numpy.zeros( len( item_users))
    matriz_user_item = []
    index_itens = dict()
    cont=0
    for item_user in item_users.keys():        
        index_itens[item_user] = cont
        cont+=1  
    print(f'Quantidade de usuarios: {len(user_item)}')    
    for user, itens in user_item.items(): # O(n^2) - no caso de um usuario ter todos os itens, na maior parte dos caso O(n)
        line_matrix = [0] * len(item_users)
        for item_nota in itens : # item usuario            
            line_matrix[ index_itens[ item_nota[0] ] ] = item_nota[1]  # index item     
        #matriz_user_item = numpy.append(line_matrix, matriz_user_item, axis=0)
        matriz_user_item.append(line_matrix)
    
    return matriz_user_item

def vocabulary(text):
    """Retorna vocabulário que compoe o texto"""
    vocab = dict()
    text = remove_accents(text)
    cont=0
    for t in text.split(' '):
        #if cont == 5000: 
        #    break; 
        #cont+=1
        if t not in stopwords and len(t) > 2:
            vocab[t] = 1
    return vocab


def remove_accents(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


if __name__ == "__main__":        
    user_item, item_user, item_content, target_user_item = leitura(sys.argv[1], sys.argv[2], sys.argv[3])  
    
    #print(len(target_user_item)); exit()
    
    matriz_user_item = matrix(user_item, item_user)  
    #matriz_user_item = numpy.array(matriz_user_item)
    #numpy.save('matriz_user_item', matriz_user_item)
    #matriz_user_item = numpy.load('matriz_user_item.npy')
    #exit()
    
    
     
    

    '''
    # metodo Rochio
    # vocabulario
    titles = []
    for k, v in item_content.items():        
        titles.append( v)
    vocab = vocabulary (' '.join(titles) )

    term_itens = []
    for term in vocab.keys(): 
        frequencia = []
        #print( len( item_content.keys()) )
        for key in item_content.keys(): 
            
            #print( item_content[key].count(term) )  
            frequencia.append(item_content[key].count(term))          
            #if term in item_content[key]:
            #    frequencia.append(1)
            #else:
            #    frequencia.append(0)                
        term_itens.append(frequencia)
    term_itens = numpy.array(term_itens)
    #print(len(term_itens))
    #numpy.save('terms_itens', term_itens)
    #exit()
    '''
    
    
    
    #term_itens = numpy.load('terms_itens.npy')       
    
    #for index_line in matriz_user_item:
    #    escore = cosine(numpy.array(matriz_user_item[index_line]), numpy.array(matriz_user_item[index_line_2])) 
    

    # indice da matriz
    index_user = dict()
    cont=0
    for user in user_item.keys():        
        index_user[user] = cont
        cont+=1  
    index_itens = dict()
    cont=0
    for item_user in item_user.keys():        
        index_itens[item_user] = cont
        cont+=1
    
    
    cont=0
    #for user, item in target_user_item.items(): 
    escreve = open('saida', 'w')
    escreve.write('UserId:ItemId,Prediction\n')

    for user, item in target_user_item:
        pred=7
        if user_item.get(user) != None:
            dados_user = statistics.median( [ float(v[1]) for v in user_item[user]] )            
            
            if item_content.get(item) != None: 
                #print(item_content[item])     
                if 'n/a' in item_content[item]['imdbvotes']:
                    pred = dados_user
                elif ',' in item_content[item]['imdbvotes'] or int(item_content[item]['imdbvotes']) > 10:                                        
                    pred = item_content[item]['imdbrating']
                else:
                    pred = dados_user
            else:
                pred = dados_user
        else:
            if item_content.get(item) != None:                
                pred = item_content[item]['imdbrating']                           

        escreve.write(f'{user}:{item},{pred}\n')
    
    
# -- backup
'''
# Rochio
cosseno = []
    for user, item in target_user_item.items(): 
        #print(  ); exit()
        #print(cont);cont+=1
        #if cont == 100: break
        #cont+=1         
        if index_user.get(user) != None:   
            dado_usuario = matriz_user_item[ index_user[user] ]   
            divisao =  int(numpy.count_nonzero( numpy.array(dado_usuario)  ))
            user_term = []               
            for t in term_itens:                                   
                temp = (numpy.dot( t, dado_usuario ) / divisao)                                           
                user_term.append( temp )
   
            if index_itens.get(item) != None:                
                index_item = index_itens[item] # index item da matriz de termos
                #print('aqui1 ', numpy.count_nonzero( numpy.array(user_term)  ))
                #print('aqui2 ', numpy.count_nonzero( term_itens[:,index_item]) )
                y_pred = cosine(numpy.array(user_term), term_itens[:,index_item] )
                cosseno.append(y_pred)
                if len(cosseno) == 1000:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots( figsize=(20, 10))
                    ax.boxplot(cosseno)
                    plt.savefig('cosseno.png', format='png')

                if y_pred == 0:
                    y_pred=7
                #y_pred = (cos.cosine_similarity( [numpy.array(user_term)], [term_itens[:,index_item]]  ))[0][0] 
            else:
                y_pred = 7
            print(y_pred)
    
'''