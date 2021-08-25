#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Sistema de recomendação filmes
import sys
import numpy
import math
import statistics
import json
import unicodedata
#import sklearn.metrics.pairwise as cos  # Calculate similarity cosine


stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def k_max_index(array, k):
    """ Return index of max values, more eficient. Example: k_max_index2([2, 4, 5, 1, 8], 2)"""
    array = numpy.array(array)
    if len(array) < k:
        k = len(array)
    indexs = numpy.argpartition(array, -k)[-k:]
    return indexs[numpy.argsort(-array[indexs])].tolist()

def cosine(x, y):
    EPSILON = 1e-07
    dot_products = numpy.dot(x, y.T)
    norm_products = numpy.linalg.norm(x) * numpy.linalg.norm(y)
    return dot_products / (norm_products + EPSILON)

def leitura(content, rating, target):
    
    """Leitura dos dados."""
    rating = open(rating); content = open(content); target = open(target)
    next(rating); next(content); next(target) # jump head
    user_item = dict(); item_content = dict(); item_user = dict(); target_user_item = dict()

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
        temp_json = json.loads("".join(line[1:]).lower())          

        if temp_json.get('actors') != None:
            item_content[line[0]] = temp_json['actors']
        else:
            item_content[line[0]] = ''
            
        
        #exit()
        #item_content[line[0]] =   line[1].split('"')[3].lower() # title
        #cotent_item[line[0]] =   json.loads("".join(line[1:]).lower())

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

def similaridade_entre_users(user_item, item_user, matriz_user_item):
    """Calcula similaridade entre usuarios"""
    users_similaridade = dict()
    keys_users =  list(user_item.keys())

    for index_line in range( len( matriz_user_item) ):
        escores_user = []
        #for index_line_2 in range(index_line+1, len( matriz_user_item)):
        for index_line_2 in range(1, len( matriz_user_item)):
            escore = cosine(numpy.array(matriz_user_item[index_line]), numpy.array(matriz_user_item[index_line_2])) 
            escores_user.append(escore)
        users_similaridade[keys_users[index_line]] = escores_user
    return users_similaridade

def similaridade_entre_users2(user_item, item_users, matriz_user_item, targets):    
    index_user = dict()
    cont=0
    for user in user_item.keys():        
        index_user[user] = cont
        cont+=1  
    index_itens = dict()
    cont=0
    for item_user in item_users.keys():        
        index_itens[item_user] = cont
        cont+=1
        
    targets = open(targets)
    next(targets) # jump head
    for line in targets:          
        escores_user = []             
        target_user_item = line.strip().split(":")
        if item_users.get(target_user_item[1] ) == None:
            continue
        
        itens_consumidos = item_users[ target_user_item[1] ]
        
        if user_item.get(target_user_item[0]) == None: # usuario novo            
            pass
        else:
            index_matrix = index_user[ target_user_item[0] ] 
            #for  index_line in range( len( matriz_user_item) ): # calcula a similaridade do alvo com os outros usuarios
            for  index_line in itens_consumidos: # calcula a similaridade do alvo com os outros usuarios
                index_line = index_user[ index_line[0] ]
                #print(index_line); exit()
                if index_line != index_matrix:
                    escore = cosine(numpy.array(matriz_user_item[index_matrix]), numpy.array(matriz_user_item[index_line]))
                    #escore = (cos.cosine_similarity( [matriz_user_item[index_matrix]], [matriz_user_item[index_line]] ))[0][0] 
                    escores_user.append(escore)
            
            k_similares_index = k_max_index(escores_user, 3) # escolhe os 3 mais similares  
            nota_final = 0
            soma_similaridade = 0
            for index_k_user in k_similares_index: # buca a nota dos mais similares
                soma_similaridade += abs(escores_user[index_k_user])
                if index_itens.get( target_user_item[1] ) == None: # item novo
                    pass
                else:                    
                    index_item = index_itens[target_user_item[1]]  # indice do item na matrix
                    nota_vizinho = matriz_user_item[index_k_user][index_item]  # nota dada pelo usuario no item
                    #contribuicao_vizinho = nota_vizinho * escores_user[index_k_user]
                    nota_final += nota_vizinho * escores_user[index_k_user]
                    #if contribuicao_vizinho != 0:
                    #    print( contribuicao_vizinho )
            if soma_similaridade == 0.0:
                nota_final=5
            else:
                #print(f'{nota_final}, {soma_similaridade}')
                nota_final = nota_final / soma_similaridade
            if nota_final==0: 
                nota_final=5
            if nota_final != 5:
                print(nota_final)

def cruzamento_dados(user_item, item_users, matriz_user_item, users_similaridade,  targets):
    index_itens = dict()
    cont=0
    for item_user in item_users.keys():        
        index_itens[item_user] = cont
        cont+=1 
    
    targets = open(targets)
    next(targets) # jump head
    for line in targets:
        target_user_item = line.strip().split(":")
        if users_similaridade.get(target_user_item[0]) == None: # usuario novo            
            pass
        else:
            k_similares = users_similaridade[target_user_item[0]]
            k_similares_index = k_max_index(k_similares, 3) # escolhe os 3 mais similares  

            for index_k_user in k_similares_index:
                if index_itens.get( target_user_item[1] ) == None: # item novo
                    pass
                else:                    
                    index_item = index_itens[target_user_item[1]]  # indice do item na matrix
                    nota_vizinho = matriz_user_item[index_k_user][index_item]  # nota dada pelo usuario no item
                    contribuicao_vizinho = nota_vizinho * k_similares[index_k_user]
                    if contribuicao_vizinho != 0:
                        print( contribuicao_vizinho )

def media(user_item, item_users, matriz_user_item, targets):
    escreve = open('saida', 'w')
    escreve.write('UserId:ItemId,Prediction\n')
    
    index_user = dict()
    cont=0
    for user in user_item.keys():        
        index_user[user] = cont
        cont+=1  
    index_itens = dict()
    cont=0
    for item_user in item_users.keys():        
        index_itens[item_user] = cont
        cont+=1
    
    targets = open(targets)
    next(targets) # jump head
    for line in targets:
        target_user_item = line.strip().split(":")        
        #if user_item.get(target_user_item[0]) == None: # usuario novo            
        if item_users.get(target_user_item[1]) == None: # usuario novo  
            escreve.write(f'{line.strip()},5\n') 
            pass
        else:
            #index_matrix = index_user[ target_user_item[0]] 
            #notas_user = [v[1] for v in user_item[target_user_item[0]]]
            notas_user = [v[1] for v in item_users[target_user_item[1]]]
            escreve.write(f'{line.strip()},{statistics.mean(notas_user)}\n')            
        
def vocabulary(text):
    """Retorna vocabulário que compoe o texto"""
    vocab = dict()
    text = remove_accents(text)
    cont=0
    for t in text.split(' '):
        if cont == 5000: 
            break; 
        cont+=1
        if t not in stopwords and len(t) > 2:
            vocab[t] = 1
    return vocab


def remove_accents(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


if __name__ == "__main__":        
    #print( numpy.dot( numpy.array([2,3]), numpy.array([0,3]) )  )
    #exit()    

    user_item, item_user, item_content, target_user_item = leitura(sys.argv[1], sys.argv[2], sys.argv[3])  
    #print(len(item_content)); print(len(item_user));     
    #exit()
    
    #matriz_user_item = matrix(user_item, item_user)  
    #matriz_user_item = numpy.array(matriz_user_item)
    #numpy.save('matriz_user_item', matriz_user_item)
    matriz_user_item = numpy.load('matriz_user_item.npy')
    #exit()
    
    
     
    

    #'''
    # vocabulario
    titles = []
    for k, v in item_content.items():        
        titles.append( v)
    vocab = vocabulary (' '.join(titles) )
    print('vocab ', len(vocab) )

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
    print(len(term_itens))
    #numpy.save('terms_itens', term_itens)
    #exit()
    #'''
    
    
    
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
            #print(user_term)
            #print(term_itens[:,index_item] )
            #print(item_content[item])

            #exit()
    #'''
    #numpy.save('user_term', numpy.array(user_term) )
            


    #print(len(term_itens[0] ))
         
        
    #print(type(numOfWords.values()))

    
    #print( term_itens)
    
    #media(user_item, item_user, matriz_user_item, sys.argv[2])
    #similaridade_entre_users2(user_item, item_user, matriz_user_item, sys.argv[2])
    #users_similaridade = similaridade_entre_users(user_item, item_user, matriz_user_item)
    #cruzamento_dados(user_item, item_user, matriz_user_item, users_similaridade, sys.argv[2])    
    #decrement_media(user_item)
    
# -- backup
'''
"""for user, itens in user_item.items():
        escore = cosine(numpy.array(matriz_user_item[index]), numpy.array(matriz_user_item[2])) 
        if escore != 0:
            #print(cosine_similarity(numpy.array(matriz_user_item[index]), numpy.array(matriz_user_item[2])))
            print(escore)
    #print(matriz_user_item)
    
        
    cont=1
    for user, itens in user_item.items():
        print(cont); cont+=1
        itens_user = []; nota_user = [];         
        for item_nota in itens:
            itens_user.append(item_nota[0])
            nota_user.append(item_nota[1])
        for index_item in range(len(list_itens)):
            if list_itens[index_item] in itens_user:                
                #matriz_user_item.append(nota_user[])
                #break
            #else:
    """       

def cosine_similarity(a, b):
    return sum([i*j for i,j in zip(a, b)])/(math.sqrt(sum([i*i for i in a]))* math.sqrt(sum([i*i for i in b])))

def matrix(user_item, item_users):
    list_itens = []
    for item_user in item_users.keys():
        list_itens.append(item_user)
    list_user = []
    user_itens = []
    for user, itens in user_item.items(): # todos os usuarios
        for index_item in range( len( list_itens ) ): # todos os itens existente do dicionario
            for index_item_user in range(len(itens)): # todos os itens do usuario
                if itens[index_item_user][0]  == list_itens[index_item]:
                    user_itens.append(item_user[1])
                    break
            if index_item_user == len(itens): # preenche a matriz com 0 para itens nao avaliados pelo usuario
                user_itens.append(0)

def decrement_media(user_item):
    """"Decrement of average"""
    print( user_item['u0026762'] )
    for k in user_item:        
        avg = 0
        for item_pred in user_item[k]:            
            avg += item_pred[1]
        #for item_pred in user_item[k]:
        #    item_pred[1] = item_pred[1] - avg
    print( user_item['u0026762'] )


    
    
# matriz termos x itens
#uniqueWords = set(vocab.keys()) 
term_itens = []
for item, conteudo in item_content.items():
    numOfWords = dict.fromkeys(uniqueWords, 0)        
    for word in conteudo.split(' '):        
        numOfWords[word] += 1
    term_itens.append( list(numOfWords.values()) )
    
'''