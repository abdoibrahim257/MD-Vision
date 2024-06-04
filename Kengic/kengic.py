from utils import *
import pandas as pd
stopwords = pd.read_csv('stopwords.csv').values

def create_graph(ngram_df,keywords,n,parents,hops): #Algorithm 3
    queue = keywords
    hop = 0
    columns = [i for i in range(1, n+1)]
    graph = {}
    edges = {}
    
    print('\nGenerating graph with keywords:' + str(keywords), "hops: ", str(hops), "parents: ", str(parents))
    
    while hop < hops and len(queue) > 0:
        current_parent = queue[0].split()
        
        # This is done to get parents of parents in case of hops > 0 (gn0 in paper)
        if len(current_parent) == 1:
            current_node = current_parent[0]
        else:
            current_node = ' '.join(current_parent[0:-1])
        
        graph_node = queue[0]
        queue = queue[1:]
        if graph_node.isdigit():
            current_hop = int(graph_node)
            if current_hop > hop:
                hop = hop + 1
            continue
        
        elif graph_node not in stopwords:
            parents_data = get_parents(n,ngram_df,graph_node, parents) #gab 23la 5 probabilities were en el kelma fe 25r el ngram   
            captions = parents_data[columns].apply(lambda x: ' '.join(x), axis=1).values #joining the n-gram into one sentence 
            if current_node not in graph.keys():
                graph[current_node] = []    

            for cap_idx, cap in enumerate(captions):
                parent_node = ' '.join(cap.split()[0:-1])
                from_prob = parents_data[cap_idx:cap_idx+1]['from_prob'].values[0]
                edges[parent_node + ':' + current_node] = from_prob
                graph[parent_node] = [current_node]

            queue += [str(hop+1)] + list(captions)
    
    return graph, edges

def top_down_traversal(graph, keywords, ngram_dfs,e_f=2): #Algorithm 3
    print('Top down traversal for graph (size:' + str(len(graph.keys())) + ') keywords:' + str(keywords))
    for first_node in graph.keys():
        for second_node in graph.keys():
            # check if they are not the same node nor the second sentence is a start sentence
            if first_node != second_node and second_node.split()[0] != '<t>':
                sentence = first_node + ' ' + second_node
                connections = check_connection(sentence, ngram_dfs)

                if connections >= e_f and second_node not in graph[first_node]:
                    graph[first_node] += [second_node]
    return graph

def traverse(graph, max_iters, keywords, n2, optimiser,ngram_dfs): #Algorithm 4
    S = set()
    Q = [[keyword] for keyword in keywords]
    qi=0
    top_n = 5
    print('Traversing graph with keywords:' + str(keywords), "max_iters: ", str(max_iters))
    bestCaption = []
    bestCaptionCost = []
    
    while len(Q) > 0 and qi < max_iters:
        topCaptions,topCosts = rank(Q,keywords,top_n,n2,optimiser,ngram_dfs)
        
        #store the top caption in another global set for backup
        if len(topCaptions) > 0:
            bestCaption,bestCaptionCost = handle_global(bestCaption,bestCaptionCost,topCaptions,topCosts,top_n)
            Q = topCaptions
        else:
            break
        
        q = topCaptions[0]
        t = q[-1]
        children = graph[t]
        children_in_q = len([child for child in children if child in q])
        if children_in_q != len(children):
            for child in children:
                if child not in q:
                    if len([word for word in q if word in keywords]) == len(keywords):
                        caption = ' '.join(q)
                        if caption not in S:
                            S.add(caption)
                    else:
                        Q.append(q + [child])
        Q.remove(q)
        # print("iteration:", qi,'/',max_iters, "Captions generated: ", len(S))
        qi+=1
    
    # print("\nRanking the captions in S")
    if len(S) == 0:
        # print("\nNo captions in S using global captions instead...\n")
        S = bestCaption
    S_splitted = [caption.split() for caption in S]
    top_S, top_S_cost = rank(S_splitted,keywords,top_n,n2,optimiser,ngram_dfs)
    
    tops_S = [' '.join(caption) for caption in top_S]

    return tops_S,top_S_cost, len(graph.keys()), qi

def get_metrics(top_captions, references):
    bleuScores = get_bleu(top_captions, references)
    return bleuScores