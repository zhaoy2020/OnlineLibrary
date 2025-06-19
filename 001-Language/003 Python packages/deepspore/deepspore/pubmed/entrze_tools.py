from Bio import Entrez
import pickle 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time 
from collections import Counter
from wordcloud import WordCloud
import networkx as nx


class NCBI:
    def __init__(self, email= None):
        Entrez.email = email                                # 配置邮箱
        self.delay = 0.34                                   # 1/3 s 

    def _search(self, terms, retmax:int= 10) -> list:
        handle = Entrez.esearch(db= 'pubmed', term= terms, retmax= retmax, sort= "relevance") # search
        records = Entrez.read(handle)                       # parse
        paper_pmid_list = records['IdList']
        handle.close()                                      # close handle
        print(f"Esearched: {len(paper_pmid_list)} papers ...")
                                           
        return paper_pmid_list

    def _fetch(self, pmids) -> list:
        time.sleep(self.delay)
        records = []
        batch_size = 10000                                  # 10000 records per batch
        for i in range(0, len(pmids), batch_size):
            sub_pmids = pmids[i:i+batch_size]               # batch to efetch
            handle = Entrez.efetch(db= 'pubmed', id= sub_pmids, rettype= 'xml', retmode= 'text')
            sub_records = Entrez.read(handle)['PubmedArticle']
            records.extend(sub_records)
            handle.close()                                  # close handle
            print(f"Efetching batch: {i+len(sub_pmids)}/{len(pmids)} ...")
            if len(sub_pmids) != len(sub_records):
                print(f"Efetching warning: {len(sub_pmids)-len(sub_records)} fail.")
            time.sleep(self.delay)                          # delay for 0.34 s    

        print(f"Efetched: {len(records)} records ...")
        return records 

    def _save_records(self, records, file_path:str):
        with open(file_path, "wb") as f:
            pickle.dump(records, f)
        print(f"Save records to {file_path} ... ")

    def _load_records(self, file_path:str):
        with open(file_path, "rb") as f:
            records = pickle.load(f)
        return records

    def _parse_pubmed_article(self, pubmed_record):
        """解析单篇 PubMed 文章记录"""
        article = pubmed_record["MedlineCitation"]["Article"]
        pubmed_data = pubmed_record["PubmedData"]
        
        # 基本信息
        parsed = {
            "pmid": str(pubmed_record["MedlineCitation"]["PMID"]),
            "title": article["ArticleTitle"],
            "journal": article["Journal"]["Title"],
            "authors": [],
            "abstract": "",
            "publication_date": "",
            "keywords": [],
            "mesh_terms": [],
            "ids": [],
            "history": {}
        }
        
        # 作者信息
        if "AuthorList" in article:
            for author in article["AuthorList"]:
                if "CollectiveName" in author:
                    parsed["authors"].append(author["CollectiveName"])
                else:
                    name_parts = []
                    if "LastName" in author:
                        name_parts.append(author["LastName"])
                    if "ForeName" in author:
                        name_parts.append(author["ForeName"])
                    elif "Initials" in author:
                        name_parts.append(author["Initials"])
                    parsed["authors"].append(" ".join(name_parts))
        
        # 摘要
        if "Abstract" in article:
            abstract_text = []
            for item in article["Abstract"]["AbstractText"]:
                if hasattr(item, "attributes") and "Label" in item.attributes:
                    abstract_text.append(f"{item.attributes['Label']}: {item}")
                else:
                    abstract_text.append(str(item))
            parsed["abstract"] = " ".join(abstract_text)
        
        # 发表日期
        pub_date = article["Journal"]["JournalIssue"]["PubDate"]
        date_parts = []
        for part in ["Year", "Month", "Day"]:
            if part in pub_date:
                date_parts.append(str(pub_date[part]))
        parsed["publication_date"] = "-".join(date_parts)
        
        # 期刊详细信息
        journal_issue = article["Journal"]["JournalIssue"]
        parsed["journal_volume"] = journal_issue.get("Volume", "")
        parsed["journal_issue"] = journal_issue.get("Issue", "")
        if "Pagination" in article:
            parsed["pages"] = article["Pagination"].get("MedlinePgn", "")
        
        # 关键词
        if "KeywordList" in pubmed_record["MedlineCitation"]:
            for kw_list in pubmed_record["MedlineCitation"]["KeywordList"]:
                parsed["keywords"].extend([str(kw) for kw in kw_list])
        
        # MeSH 术语
        if "MeshHeadingList" in pubmed_record["MedlineCitation"]:
            for mesh in pubmed_record["MedlineCitation"]["MeshHeadingList"]:
                parsed["mesh_terms"].append(str(mesh["DescriptorName"]))
        
        # 文章ID (DOI等)
        for article_id in pubmed_data["ArticleIdList"]:
            parsed["ids"].append({
                "type": article_id.attributes["IdType"],
                "id": str(article_id)
            })
        
        # 文章历史
        # for event in pubmed_data["History"]:
        #     parsed["history"][event["PubStatus"]] = {
        #         "year": event["Year"],
        #         "month": event["Month"],
        #         "day": event["Day"]
        #     }
        
        return parsed

    def _parse(self, records):
        papers = []
        
        with tqdm(records, desc= "Parseing") as pbar:
            for record in pbar:
                paper = {
                    'file_path': '',
                    'year': '', 
                    'journal': '',
                    'doi': '',
                    "sections": {
                        'title': '',
                        'authors': [],
                        'abstract': '',
                        'keywords': [],
                        'mesh_terms': [],
                    },
                    "full_paper": ''
                }
                try:
                    parsed = self._parse_pubmed_article(pubmed_record= record)
                    if parsed:
                        paper['file_path'] = parsed['pmid']
                        paper['year'] = parsed['publication_date'].split('-')[0]
                        paper['journal'] = parsed['journal']
                        for i in parsed['ids']:
                            if i['type'] == 'doi':
                                paper['doi'] = i['id']
                        paper['sections']['authors'] = parsed['authors']
                        paper["sections"]['title'] = parsed['title']
                        paper["sections"]['abstract'] = parsed['abstract']
                        paper["sections"]['keywords'] = parsed['keywords']
                        paper["sections"]['mesh_terms'] = parsed['mesh_terms']
                        paper["full_paper"] = parsed
                        papers.append(paper)                   
                except:
                    print("Error in parsing record: ", record)
                    continue
                
        return papers
    
    def get_tiab(self, terms:str= None, retmax:int= 10, pubmed_records_path:str= "pubmed_records.pkl"):
        pmids = self._search(terms= terms, retmax= retmax)
        records = self._fetch(pmids= pmids)
        self._save_records(records= records, file_path= pubmed_records_path)

    def parse_tiab(self, records_path:str= None) -> list:
        # Load records from file
        records = self._load_records(file_path= records_path)
        # Parse records to get papers
        papers = self._parse(records= records)
        return papers

    def _publication_trend(self, papers: list):
         # Visualize the papers
        years = [paper['year'] for paper in papers]
        ## counter
        years = Counter(years)
        ## sorted
        years = {k:v for k, v in sorted(years.items(), reverse= False)}
        ## draw picture
        
        plt.bar(years.keys(), years.values())
        plt.xticks(rotation= 45)
        plt.xlabel('Year')
        plt.ylabel('Number of Papers')
        plt.title('Publication Trend')

    def _top_journal(self, papers: list):
        TOP = 10
        journals = [paper['journal'] for paper in papers if paper['journal'] is not None]
        journals = Counter(journals).most_common(TOP)
        plt.barh([i[0] for i in journals], [i[1] for i in journals])
        plt.xlabel('Number of Publications')
        plt.title('Top 10 Journals')

    def _word_cloud(self, papers:list):
        text = [paper['sections']['title']+
                paper['sections']['abstract']+
                " ".join(paper["sections"]['keywords'])+
                " ".join(paper["sections"]['mesh_terms']) 
                for paper in papers]
        text = " ".join(text)
        wordcloud = WordCloud(width= 800, height= 500, background_color= 'white').generate(text)
        plt.imshow(wordcloud, interpolation= 'bilinear', 
                   aspect='auto'                                    # # 自动适配画布比例 
                   )
        plt.axis('off')
        plt.title('Common Themes')

    def _network_analysis(self, papers:list):
        G = nx.Graph()  # 创建空网络
        
        # 构建合作网络(共同作者视为连接)
        for paper in papers[:10]:  # 限制文献数量
            authors = paper['sections']['authors']
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

    def visualize(self, papers:list, figsize:tuple=(10, 10)):
        nrows = 2
        ncols = 2
        plt.figure(figsize= figsize)

        # Visualize the publications by year (Publication Trend)
        plt.subplot(nrows, ncols, 1)
        self._publication_trend(papers= papers)

        # Top 10 journal
        plt.subplot(nrows, ncols, 2)
        self._top_journal(papers= papers)

        # WordCloud
        plt.subplot(nrows, ncols, 3)
        self._word_cloud(papers= papers)

        # Network
        plt.subplot(nrows, ncols, 4)
        self._network_analysis(papers= papers)


        plt.tight_layout()


def ncbi_pubmed_analysis(email:str):
    '''
    利用Biopython的Entrze检索和下载文献相关信息（题目、摘要、DOI、关键词等）并进行可视化。
    - get_tiab: 通过eseaerch和efetct将下载的记录保存成records.pickle格式；
    - parse_tiab: 解析records解析为papers格式（详间上述）。
    - visualize: 可视化为四个子图，分别为文献趋势、TOP10杂志、研究主题词云、作者网络图。
    '''

     # 实例化为对象
    pubmed = NCBI(email= "zhao_sy@126.com")

    # 检索并保存结果
    keywords = "('microbiome'[tiab] AND 'plant'[tiab]) AND ('2010/01'[dp] : '2025/03'[dp])" 
    journal = ""
    terms = keywords + journal 
    pubmed.get_tiab(terms= terms, retmax= 50500, pubmed_records_path= './cache/records.pkl')

    # 分析结果
    papers = pubmed.parse_tiab(records_path= './cache/records.pkl')
    
    # 可视化结果
    # %config InlineBackend.figure_format = 'svg'
    factor = 1.3
    pubmed.visualize(papers= papers, figsize=(8*factor, 5*factor))


from Bio import Entrez
import os
import time
from bs4 import BeautifulSoup
import requests
from urllib.error import HTTPError
from typing import List, Dict, Optional

class PubMedFullTextDownloader:
    def __init__(self, email: str, api_key: str = None):
        """
        初始化PubMed全文下载器
        
        参数:
            email (str): 必须提供有效的邮箱地址(NCBI要求)
            api_key (str, optional): NCBI API密钥(如果有)
        """
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        self.delay = 0.34  # 遵守NCBI每秒不超过3次请求的限制
    
    def _check_fulltext_availability(self, pmid: str) -> Optional[str]:
        """检查指定PMID是否有全文可用"""
        try:
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
            record = Entrez.read(handle)
            handle.close()
            
            if record and record[0]["LinkSetDb"]:
                pmc_id = record[0]["LinkSetDb"][0]["Link"][0]["Id"]
                return f"PMC{pmc_id}"
            return None
        except Exception as e:
            print(f"检查PMID {pmid}全文可用性时出错: {str(e)}")
            return None
    
    def _download_pmc_xml(self, pmc_id: str, save_path: str) -> bool:
        """下载PMC XML格式全文"""
        try:
            handle = Entrez.efetch(db="pmc", id=pmc_id, retmode="xml")
            with open(save_path, "wb") as f:
                f.write(handle.read())
            handle.close()
            return True
        except Exception as e:
            print(f"下载PMC {pmc_id} XML全文失败: {str(e)}")
            return False
    
    def _download_pdf(self, pmc_id: str, save_path: str) -> bool:
        """尝试下载PDF全文"""
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
        try:
            response = requests.get(pdf_url, timeout=10)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                return True
            return False
        except Exception as e:
            print(f"下载PMC {pmc_id} PDF失败: {str(e)}")
            return False
    
    def _parse_pubmed_record(self, pmid: str) -> Optional[Dict]:
        """获取PubMed记录的基本信息"""
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
            record = Entrez.read(handle)
            handle.close()
            
            if not record["PubmedArticle"]:
                return None
                
            article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]
            pubmed_data = record["PubmedArticle"][0]["PubmedData"]
            
            # 提取基本信息
            result = {
                "pmid": pmid,
                "title": article["ArticleTitle"],
                "journal": article["Journal"]["Title"],
                "authors": [],
                "abstract": "",
                "publication_date": "",
                "doi": None,
                "pmcid": None,
                "has_fulltext": False
            }
            
            # 作者信息
            if "AuthorList" in article:
                for author in article["AuthorList"]:
                    if "CollectiveName" in author:
                        result["authors"].append(author["CollectiveName"])
                    else:
                        name_parts = []
                        if "LastName" in author:
                            name_parts.append(author["LastName"])
                        if "ForeName" in author:
                            name_parts.append(author["ForeName"])
                        elif "Initials" in author:
                            name_parts.append(author["Initials"])
                        result["authors"].append(" ".join(name_parts))
            
            # 摘要
            if "Abstract" in article:
                abstract_text = []
                for item in article["Abstract"]["AbstractText"]:
                    if hasattr(item, "attributes") and "Label" in item.attributes:
                        abstract_text.append(f"{item.attributes['Label']}: {item}")
                    else:
                        abstract_text.append(str(item))
                result["abstract"] = " ".join(abstract_text)
            
            # 发表日期
            pub_date = article["Journal"]["JournalIssue"]["PubDate"]
            date_parts = []
            for part in ["Year", "Month", "Day"]:
                if part in pub_date:
                    date_parts.append(str(pub_date[part]))
            result["publication_date"] = "-".join(date_parts)
            
            # DOI和PMCID
            for article_id in pubmed_data["ArticleIdList"]:
                if article_id.attributes["IdType"] == "doi":
                    result["doi"] = str(article_id)
                elif article_id.attributes["IdType"] == "pmc":
                    result["pmcid"] = str(article_id)
                    result["has_fulltext"] = True
            
            return result
            
        except Exception as e:
            print(f"解析PMID {pmid}记录失败: {str(e)}")
            return None
    
    def _parse_pmc_xml(self, xml_file: str) -> Dict:
        """解析PMC XML全文"""
        with open(xml_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "xml")
        
        # 提取标题
        title = soup.article.title.get_text() if soup.article and soup.article.title else ""
        
        # 提取作者
        authors = []
        if soup.article:
            for author in soup.find_all("contrib"):
                if author.get("contrib-type") == "author":
                    surname = author.find("surname").get_text() if author.find("surname") else ""
                    given_names = author.find("given-names").get_text() if author.find("given-names") else ""
                    authors.append(f"{given_names} {surname}")
        
        # 提取摘要
        abstract = ""
        if soup.abstract:
            abstract = "\n".join(p.get_text() for p in soup.abstract.find_all("p"))
        
        # 提取正文文本
        body_text = []
        if soup.body:
            for sec in soup.body.find_all("sec"):
                title = sec.title.get_text() if sec.title else ""
                paragraphs = [p.get_text() for p in sec.find_all("p")]
                body_text.append({"section": title, "content": "\n".join(paragraphs)})
        
        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "body": body_text
        }
    
    def download_pmid(self, pmid: str, output_dir: str) -> Dict:
        """
        下载单篇PMID的文献信息
        
        参数:
            pmid (str): PubMed ID
            output_dir (str): 输出目录路径
            
        返回:
            dict: 包含文献信息和下载结果的字典
        """
        os.makedirs(output_dir, exist_ok=True)
        result = {
            "pmid": pmid,
            "metadata": None,
            "xml_path": None,
            "pdf_path": None,
            "success": False,
            "message": ""
        }
        
        try:
            # 获取文献元数据
            metadata = self._parse_pubmed_record(pmid)
            if not metadata:
                result["message"] = "无法获取文献元数据"
                return result
                
            result["metadata"] = metadata
            
            # 检查全文可用性
            pmc_id = metadata["pmcid"] or self._check_fulltext_availability(pmid)
            
            if pmc_id:
                metadata["pmcid"] = pmc_id
                metadata["has_fulltext"] = True
                
                # 下载XML全文
                xml_path = os.path.join(output_dir, f"{pmid}.xml")
                if self._download_pmc_xml(pmc_id, xml_path):
                    result["xml_path"] = xml_path
                    # 解析XML内容
                    metadata.update(self._parse_pmc_xml(xml_path))
                
                # 尝试下载PDF
                pdf_path = os.path.join(output_dir, f"{pmid}.pdf")
                if self._download_pdf(pmc_id, pdf_path):
                    result["pdf_path"] = pdf_path
                
                result["success"] = True
                result["message"] = "全文下载成功" if result["xml_path"] or result["pdf_path"] else "只有摘要可用"
            else:
                result["message"] = "无全文可用，仅保存元数据"
                result["success"] = True
            
            return result
            
        except Exception as e:
            result["message"] = f"处理PMID {pmid}时出错: {str(e)}"
            return result
        finally:
            time.sleep(self.delay)  # 遵守API请求频率限制
    
    def batch_download(self, pmid_list: List[str], output_dir: str) -> List[Dict]:
        """
        批量下载多篇PMID的文献
        
        参数:
            pmid_list (List[str]): PMID列表
            output_dir (str): 输出目录路径
            
        返回:
            List[Dict]: 每篇文献的下载结果列表
        """
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, pmid in enumerate(pmid_list):
            print(f"处理中 [{i+1}/{len(pmid_list)}]: PMID {pmid}")
            result = self.download_pmid(pmid, output_dir)
            results.append(result)
            
            # 打印简要结果
            status = "成功" if result["success"] else "失败"
            fulltext = "有全文" if result["metadata"]["has_fulltext"] else "无全文"
            print(f"状态: {status} | 全文: {fulltext} | 消息: {result['message']}")
        
        return results


def download_fulltext(email:str, pmids:list, output_dir:str):
    # 1. 初始化下载器(替换为您的邮箱)
    downloader = PubMedFullTextDownloader(email= email)
    
    # 2. 准备PMID列表(替换为实际PMID)
    # pmids = [
    #     "32476065",  # 有PMC全文的示例
    #     "32139689",  # 另一个有PMC全文的示例
    #     "12345678"   # 不存在的PMID用于测试错误处理
    # ]
    
    # 3. 指定输出目录
    # output_directory = "pubmed_downloads"
    
    # 4. 批量下载
    results = downloader.batch_download(pmids, output_dir)
    
    # 5. 打印摘要报告
    print("\n下载摘要:")
    success_count = sum(1 for r in results if r["success"])
    fulltext_count = sum(1 for r in results if r["metadata"]["has_fulltext"])
    print(f"处理文献数: {len(results)}")
    print(f"成功下载: {success_count}")
    print(f"获取全文: {fulltext_count}")