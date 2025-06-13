# 导入必要的库
from Bio import Entrez  # Biopython中用于访问NCBI Entrez数据库的模块
from Bio.Entrez import esearch, efetch, read  # Entrez的搜索、获取和读取功能
import matplotlib.pyplot as plt  # 数据可视化
from collections import defaultdict  # 提供默认值字典
from wordcloud import WordCloud  # 生成词云
import re  # 正则表达式
import time  # 时间相关操作
from sklearn.feature_extraction.text import TfidfVectorizer  # 文本特征提取
from sklearn.decomposition import LatentDirichletAllocation  # 主题模型(LDA)
import networkx as nx  # 复杂网络分析


class PubMedAnalyzer:
    """
    PubMed文献分析工具类
    
    功能：
    - 搜索PubMed文献
    - 提取文献元数据(标题、作者、期刊、发表日期等)
    - 分析文献趋势和主题
    - 可视化分析结果
    
    使用示例：
    >>> analyzer = PubMedAnalyzer(email="your@email.com")
    >>> results = analyzer.analyze_publications("CRISPR", max_results=100)
    >>> analyzer.visualize_trends(results, "CRISPR")
    """
    
    def __init__(self, email, api_key= None):
        """
        初始化PubMed分析工具
        
        参数:
            email (str): 必须提供有效的邮箱地址，NCBI要求用于API访问
            api_key (str, optional): 如果有PubMed API密钥可以提高请求限制
            
        属性:
            delay (float): API请求之间的延迟(秒)，遵守NCBI的每秒最多3次请求的限制
        """
        Entrez.email = email  # 设置NCBI API的邮箱
        if api_key:
            Entrez.api_key = api_key  # 设置API密钥(如果有)
        self.delay = 0.34  # 延迟时间(1/3秒)确保不超过API限制
        
    def search_pubmed(self, keyword, max_results=500, mindate=None, maxdate=None):
        """
        在PubMed中搜索文献
        
        参数:
            keyword (str): 搜索关键词，可以是布尔表达式
            max_results (int): 返回的最大结果数(默认500)
            mindate (str): 最小日期(格式: YYYY/MM/DD或YYYY或YYYY/MM)
            maxdate (str): 最大日期
            
        返回:
            list: 匹配的PubMed ID列表
            
        注意:
            - 自动添加延迟以避免超过API请求限制
            - 可以按日期范围筛选结果
        """
        time.sleep(self.delay)  # 遵守API请求频率限制
        
        term = keyword
        # 如果有日期限制，添加到搜索词中
        if mindate or maxdate:
            term += f" AND ({self._format_date_range(mindate, maxdate)})"
            
        # 执行搜索
        handle = esearch(db= "pubmed", term= term, retmax= max_results, sort= "relevance")
        record = read(handle)  # 读取返回的XML结果
        handle.close()
        return record["IdList"]  # 返回PMID列表
    
    def fetch_publication_details(self, pubmed_ids):
        """
        获取文献的详细元数据
        
        参数:
            pubmed_ids (list): PubMed ID列表
            
        返回:
            list: 包含每篇文献完整元数据的字典列表
            
        注意:
            - 分批获取(每批100个)以避免大请求
            - 自动添加延迟遵守API限制
        """
        time.sleep(self.delay)  # API请求延迟
        
        batch_size = 100  # 每批处理100篇文献
        records = []
        for i in range(0, len(pubmed_ids), batch_size):
            batch = pubmed_ids[i:i+batch_size]
            handle = efetch(db= "pubmed", id= batch, retmode= "xml")
            records.extend(read(handle)['PubmedArticle'])  # 解析XML
            handle.close()
            time.sleep(self.delay)  # 批次间延迟
            
        return records
    
    def analyze_publications(self, keyword, max_results=500, mindate=None, maxdate=None):
        """
        主分析方法：搜索文献并提取分析所需数据
        
        参数:
            keyword (str): 搜索关键词
            max_results (int): 最大返回结果数
            mindate (str): 最小日期
            maxdate (str): 最大日期
            
        返回:
            dict: 包含以下键的分析结果字典:
                - year_counts: 每年文献计数
                - journal_counts: 期刊分布
                - authors: 作者列表
                - titles: 标题列表
                - abstracts: 摘要列表
                - keywords: 关键词列表
                - pub_details: 每篇文献的完整详情
                
        注意:
            - 处理过程中会跳过格式不正确的记录
            - 自动处理多种PubMed数据格式
        """
        # 1. 搜索文献获取PMID列表
        pubmed_ids = self.search_pubmed(keyword, max_results, mindate, maxdate)
        print(f"esearch 找到{len(pubmed_ids)} 篇文献 ...")
        
        # 2. 获取详细文献数据
        records = self.fetch_publication_details(pubmed_ids)
        
        # 3. 初始化结果字典
        results = {
            'year_counts': defaultdict(int),  # 按年计数
            'journal_counts': defaultdict(int),  # 按期刊计数
            'authors': [],  # 所有作者
            'titles': [],  # 所有标题
            'abstracts': [],  # 所有摘要
            'keywords': [],  # 所有关键词
            'pub_details': []  # 每篇文献的完整详情
        }
        
        # 4. 处理每篇文献
        for record in records:
            try:
                article = record['MedlineCitation']['Article']
                journal_info = article['Journal']
                
                # 4.1 获取发表日期(处理多种可能的日期格式)
                pub_date = None
                if 'ArticleDate' in article and len(article['ArticleDate']) > 0:
                    pub_date = article['ArticleDate'][0]
                elif 'JournalIssue' in journal_info and 'PubDate' in journal_info['JournalIssue']:
                    pub_date = journal_info['JournalIssue']['PubDate']
                
                # 4.2 提取年份
                year = self._extract_year_from_date(pub_date) if pub_date else None
                if year:
                    results['year_counts'][year] += 1
                
                # 4.3 提取期刊信息
                journal = journal_info.get('Title', 'Unknown Journal')
                results['journal_counts'][journal] += 1
                
                # 4.4 提取作者信息(处理多种作者格式)
                author_list = []
                if 'AuthorList' in article:
                    for author in article['AuthorList']:
                        if 'LastName' in author and 'ForeName' in author:
                            author_list.append(f"{author['LastName']} {author['ForeName']}")
                results['authors'].extend(author_list)
                
                # 4.5 提取标题(处理多种标题格式)
                title = article.get('ArticleTitle', 'No Title')
                if isinstance(title, dict) and 'String' in title:
                    title = title['String']
                results['titles'].append(str(title))
                
                # 4.6 提取摘要(处理多种摘要格式)
                abstract_text = ""
                if 'Abstract' in article:
                    abstract = article['Abstract']
                    if 'AbstractText' in abstract:
                        if isinstance(abstract['AbstractText'], list):
                            abstract_text = ' '.join([
                                str(item) if not isinstance(item, dict) else 
                                item.get('String', '') if 'String' in item else 
                                item.get('str', '') 
                                for item in abstract['AbstractText']
                            ])
                        elif isinstance(abstract['AbstractText'], str):
                            abstract_text = abstract['AbstractText']
                        elif isinstance(abstract['AbstractText'], dict):
                            abstract_text = abstract['AbstractText'].get('String', '')
                results['abstracts'].append(abstract_text)
                
                # 4.7 提取关键词(处理多种关键词格式)
                current_keywords = []
                if 'KeywordList' in record['MedlineCitation']:
                    for kw_list in record['MedlineCitation']['KeywordList']:
                        if isinstance(kw_list, list):
                            for kw in kw_list:
                                if isinstance(kw, str):
                                    current_keywords.append(kw)
                                elif isinstance(kw, dict) and 'String' in kw:
                                    current_keywords.append(kw['String'])
                results['keywords'].extend(current_keywords)
                
                # 4.8 保存完整文献详情
                results['pub_details'].append({
                    'pmid': str(record['MedlineCitation']['PMID']),
                    'title': str(title),
                    'authors': author_list,
                    'journal': str(journal),
                    'year': year,
                    'abstract': abstract_text,
                    'keywords': current_keywords
                })
            
            except Exception as e:
                print(f"处理文献时出错: {str(e)}")
                continue  # 跳过有问题的文献
        
        return results
    
    def visualize_trends(self, results, keyword):
        """
        可视化分析结果
        
        参数:
            results (dict): analyze_publications返回的结果字典
            keyword (str): 搜索关键词(用于标题)
            
        生成4个子图:
            1. 年度发表趋势(柱状图)
            2. 期刊分布(横向条形图)
            3. 主题词云
            4. 作者合作网络
        """
        plt.figure(figsize=(15, 10))  # 创建大图
        
        # 子图1: 年度发表趋势
        plt.subplot(2, 2, 1)
        sorted_years = sorted(results['year_counts'].items())  # 按年份排序
        years = [y[0] for y in sorted_years]
        counts = [y[1] for y in sorted_years]
        plt.bar(years, counts)
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        # plt.title(f'Publications on "{keyword}" by Year')
        plt.title("Publication Trend")
        plt.xticks(rotation= 45)  # 旋转x轴标签
        
        # 子图2: 期刊分布(前10)
        plt.subplot(2, 2, 2)
        journals = sorted(results['journal_counts'].items(), key= lambda x: x[1], reverse= True)[:10]  # 取前10
        plt.barh([j[0] for j in journals], [j[1] for j in journals])  # 横向条形图
        plt.xlabel('Number of Publications')
        plt.title('Top 10 Journals')
        
        # 子图3: 主题词云
        plt.subplot(2, 2, 3)
        text = ' '.join(results['titles'] + results['abstracts'] + results['keywords'])
        wordcloud = WordCloud(width= 800, height= 400, background_color= 'white').generate(text)
        plt.imshow(wordcloud, interpolation= 'bilinear')
        plt.axis('off')
        plt.title('Common Themes')
        
        # 子图4: 作者合作网络
        plt.subplot(2, 2, 4)
        self._plot_author_network(results['pub_details'])
        
        plt.tight_layout()  # 调整子图间距
        plt.show()
        
        # 如果有足够摘要，进行LDA主题分析
        if len(results['abstracts']) > 10:
            self._analyze_topics(results['abstracts'])
    
    def _format_date_range(self, mindate, maxdate):
        """
        内部方法：格式化日期范围查询字符串
        
        参数:
            mindate (str): 最小日期
            maxdate (str): 最大日期
            
        返回:
            str: 格式化的日期范围字符串，如"2020[PDAT]:2023[PDAT]"
        """
        date_range = []
        if mindate:
            date_range.append(f"{mindate}[PDAT]")  # PubMed日期字段标识
        if maxdate:
            date_range.append(f"{maxdate}[PDAT]")
        return ":".join(date_range) if len(date_range) > 1 else date_range[0]
    
    def _extract_year_from_date(self, date):
        """
        内部方法：从日期信息中提取年份
        
        处理多种日期格式:
            - 字典格式: {'Year': '2023', ...}
            - 字符串格式: "2023 Jan"或"2023-01-15"
            
        参数:
            date (dict|str): 日期信息
            
        返回:
            int: 提取的年份或None(如果无法提取)
        """
        if isinstance(date, dict):
            if 'Year' in date:
                return int(date['Year'])
        elif isinstance(date, str):
            match = re.search(r'\d{4}', date)  # 查找4位数字的年份
            if match:
                return int(match.group())
        return None
    
    def _plot_author_network(self, pub_details):
        """
        内部方法：绘制作者合作网络图
        
        参数:
            pub_details (list): 文献详情列表
            
        注意:
            - 只分析前100篇文献避免图太复杂
            - 只显示有合作的作者(度>1)
        """
        G = nx.Graph()  # 创建空网络
        
        # 构建合作网络(共同作者视为连接)
        for pub in pub_details[:10]:  # 限制文献数量
            authors = pub['authors']
            for author in authors:
                G.add_node(author)  # 添加节点
            # 添加作者间的边(合作)
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]['weight'] += 1  # 增加合作权重
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)  # 新合作
        
        # 过滤孤立节点(无合作的作者)
        nodes_to_keep = [n for n in G.nodes() if G.degree(n) > 1]
        G = G.subgraph(nodes_to_keep)
        
        if len(G.nodes()) == 0:
            print("合作网络数据不足")
            return
        
        # 绘制网络
        pos = nx.spring_layout(G, k= 0.5, iterations= 50, seed= 10)   # 网络布局算法，模拟弹簧斥力的布局算法，使节点分布均匀。
        nx.draw_networkx_nodes(G, pos, node_size= 50)       # 绘制节点（大小50）
        nx.draw_networkx_edges(G, pos, alpha= 0.2)          # 绘制边（透明度0.2）
        nx.draw_networkx_labels(G, pos, font_size= 8)       # 绘制节点标签（字号8）
        plt.title("Author Collaboration Network")
        plt.axis('off')
    
    def _analyze_topics(self, abstracts, n_topics= 5, n_top_words= 10):
        """
        内部方法：使用LDA进行主题分析
        
        参数:
            abstracts (list): 摘要文本列表
            n_topics (int): 要提取的主题数量
            n_top_words (int): 每个主题显示的关键词数量
            
        注意:
            - 使用TF-IDF向量化文本
            - 使用LDA进行主题建模
        """
        # 过滤空摘要
        filtered_abstracts = [ab for ab in abstracts if ab.strip()]
        
        if len(filtered_abstracts) < n_topics:
            print(f"需要至少 {n_topics} 篇有摘要的文献进行主题分析")
            return
        
        # 1. 文本向量化(TF-IDF)
        tfidf = TfidfVectorizer(max_df= 0.95, min_df= 2, stop_words= 'english')
        tfidf_matrix = tfidf.fit_transform(filtered_abstracts)
        
        # 2. 训练LDA模型
        lda = LatentDirichletAllocation(n_components= n_topics, random_state= 42)
        lda.fit(tfidf_matrix)
        
        # 3. 显示每个主题的关键词
        feature_names = tfidf.get_feature_names_out()
        
        print("\n主题模型分析结果:")
        for topic_idx, topic in enumerate(lda.components_):
            # 获取权重最高的关键词
            top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
            print(f"主题 #{topic_idx + 1}: {', '.join(top_features)}")


def analysis(email:str, keyword: str, max_results: int = 1000, mindate: str = "2010", maxdate: str = "2025"):
    '''
    分析PubMed文献:

    演示：
    >>> terms = ''("deep learning"[tiab] OR "machine learning"[tiab] OR "artificial intelligence"[tiab] OR "AI"[tiab]) ''
    >>> analysis(email= 'demo@126.com', keyword= terms, max_results= 1000, mindate= "2010", maxdate= "2025")
    '''

    # 1. 初始化分析器(必须提供有效邮箱)
    analyzer = PubMedAnalyzer(email= email)
    
    # 2. 设置搜索参数
    
    # 3. 执行分析
    print(f"开始分析关键词: {keyword}")
    results = analyzer.analyze_publications(keyword, max_results, mindate, maxdate)
    
    # 4. 可视化结果
    analyzer.visualize_trends(results, keyword)
    
    # 5. 显示统计摘要
    print("\n分析摘要:")
    print(f"时间范围: {min(results['year_counts'].keys())}-{max(results['year_counts'].keys())}")
    print(f"总文献数: {sum(results['year_counts'].values())}")
    print(f"涉及期刊数: {len(results['journal_counts'])}")
    print(f"涉及作者数: {len(set(results['authors']))}")
    
    # 显示最高产期刊
    top_journal = max(results['journal_counts'].items(), key= lambda x: x[1])
    print(f"最高产期刊: {top_journal[0]} ({top_journal[1]}篇)")
