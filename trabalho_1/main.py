#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Sistema de recomendação filmes
import sys
import numpy
#import math
import statistics

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

def leitura(rating):
    """Leitura dos dados."""
    rating = open(rating)    
    next(rating) # jump head
    user_item = dict()
    item_user = dict()
    cont = 0
    for line in rating: # criar dicionario que tem as relacoes dos usuario, itens e notas
        if cont==3000: break
        cont+=1
        line = line.strip().split(",")
        user_item_line = line[0].split(":")
        if user_item.get(user_item_line[0]) == None: 
            user_item[user_item_line[0]] = []
        user_item[user_item_line[0]].append( (user_item_line[1], int(line[1]) )) 
        
        if item_user.get(user_item_line[1]) == None:
            item_user[user_item_line[1]] = []
        item_user[user_item_line[1]].append( (user_item_line[0], int(line[1]) ) )
    return user_item, item_user    

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
            for index_k_user in k_similares_index: # buca a nota dos mais similares
                if index_itens.get( target_user_item[1] ) == None: # item novo
                    pass
                else:                    
                    index_item = index_itens[target_user_item[1]]  # indice do item na matrix
                    nota_vizinho = matriz_user_item[index_k_user][index_item]  # nota dada pelo usuario no item
                    contribuicao_vizinho = nota_vizinho# * k_similares[index_k_user]
                    if contribuicao_vizinho != 0:
                        print( contribuicao_vizinho )

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
        


if __name__ == "__main__":
    user_item, item_user = leitura(sys.argv[1])    
    matriz_user_item = matrix(user_item, item_user)  
    #media(user_item, item_user, matriz_user_item, sys.argv[2])
    similaridade_entre_users2(user_item, item_user, matriz_user_item, sys.argv[2])
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



'''