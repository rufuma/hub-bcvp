from py2neo import Graph, Node, Relationship, NodeMatcher

# ===================== 1. 连接Neo4j数据库 =====================
graph = Graph(
    "bolt://localhost:7687",
    auth=("neo4j", "xxx")
)

graph.run("MATCH (n) DETACH DELETE n")

# ===================== 2. 创建知识图谱节点（实体） =====================
# 节点1：周杰伦（人物节点，包含属性）
jay = Node(
    "Person",  # 节点标签（类型）
    name="周杰伦",
    birthday="1979-01-18",
    nationality="中国",
    occupation="歌手、词曲作者、演员、导演"
)
graph.create(jay)

# 节点2：《七里香》（作品节点）
album = Node(
    "Album",  # 节点标签
    name="七里香",
    release_year=2004,
    type="音乐专辑"
)
graph.create(album)

# 节点3：《头文字D》（作品节点）
movie = Node(
    "Movie",  # 节点标签
    name="头文字D",
    release_year=2005,
    type="电影"
)
graph.create(movie)

# 节点4：昆凌（人物节点）
hannah = Node(
    "Person",
    name="昆凌",
    birthday="1993-08-12",
    nationality="中国台湾",
    occupation="模特、演员"
)
graph.create(hannah)

# ===================== 3. 创建关系（三元组） =====================
# 三元组1：周杰伦 - 发行 - 七里香
rel1 = Relationship(jay, "发行", album, role="主唱/词曲创作")
graph.create(rel1)

# 三元组2：周杰伦 - 参演 - 头文字D
rel2 = Relationship(jay, "参演", movie, role="男主角（藤原拓海）")
graph.create(rel2)

# 三元组3：周杰伦 - 配偶 - 昆凌
rel3 = Relationship(jay, "配偶", hannah)
graph.create(rel3)

# ===================== 4. 知识图谱问答查询 =====================
print("===== 问答查询结果 =====")

# 查询1：查询周杰伦的基本信息
query1 = """
MATCH (p:Person {name: '周杰伦'})
RETURN p.name AS 姓名, p.birthday AS 生日, p.nationality AS 国籍, p.occupation AS 职业
"""
result1 = graph.run(query1).data()
print("1. 周杰伦的基本信息：", result1)

# 查询2：查询周杰伦发行的专辑
query2 = """
MATCH (p:Person {name: '周杰伦'})-[r:发行]->(a:Album)
RETURN p.name AS 歌手, a.name AS 专辑名, a.release_year AS 发行年份, r.role AS 角色
"""
result2 = graph.run(query2).data()
print("2. 周杰伦发行的专辑：", result2)

# 查询3：查询周杰伦的配偶信息
query3 = """
MATCH (p1:Person {name: '周杰伦'})-[r:配偶]->(p2:Person)
RETURN p1.name AS 本人, r.type AS 关系, p2.name AS 配偶, p2.birthday AS 配偶生日
"""
result3 = graph.run(query3).data()
print("3. 周杰伦的配偶信息：", result3)

# 查询4：模糊查询所有和周杰伦相关的实体
query4 = """
MATCH (p:Person {name: '周杰伦'})-[r]-(n)
RETURN p.name AS 中心实体, type(r) AS 关系类型, n.name AS 关联实体, labels(n) AS 实体类型
"""
result4 = graph.run(query4).data()
print("4. 所有和周杰伦相关的实体：", result4)
