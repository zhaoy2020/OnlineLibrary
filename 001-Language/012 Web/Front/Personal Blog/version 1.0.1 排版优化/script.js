const papers = [
  {
    title: "Efficient transformation and genome editing in a nondomesticated, biocontrol strain, Bacillus subtilis GLB191",
    authors: "Yu Zhao, Zhenshuo Wang, Qian Wang, Bing Wang, Xiaoning Bian, Qingchao Zeng, Daowan Lai, Qi Wang, Yan Li",
    year: 2024,
    topic: ["Molecular Biology", "Plant Microecology"],
    img: "https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs42483-024-00287-0/MediaObjects/42483_2024_287_Fig3_HTML.png?as=webp",
    link: "https://link.springer.com/article/10.1186/s42483-024-00287-0"
  },
  {
    title: "Kiwifruit resistance to gray mold is enhanced by yeast-induced modulation of the endophytic microbiome",
    authors: "Qinhong Liao, Yu Zhao, Zhenshuo Wang, Longfeng Yu, Qiqian Su, Jiaoqian Li, Anran Yuan, Junkui Wang, Dawei Tian, Chenglin Lin, Xiaoya Huang, Wenhua Li, Zhiqiang Sun, Qi Wang, Jia Liu",
    year: 2024,
    topic: ["Plant Microecology"],
    img: "https://ars.els-cdn.com/content/image/1-s2.0-S004896972403256X-ga1.jpg",
    link: "https://www.sciencedirect.com/science/article/pii/S004896972403256X"
  },
  {
    title: "A microbial consortium‐based product promotes potato yield by recruiting rhizosphere bacteria involved in nitrogen and carbon metabolisms",
    authors: "Zhenshuo Wang, Yan Li, Yu Zhao, Lubo Zhuang, Yue Yu, Mengyao Wang, Jia Liu, Qi Wang",
    year: 2021,
    topic: ["Plant Microecology"],
    img: "https://enviromicro-journals.onlinelibrary.wiley.com/cms/asset/37993e73-2836-4dbf-928b-603deb8a793a/mbt213876-toc-0001-m.jpg",
    link: "https://enviromicro-journals.onlinelibrary.wiley.com/doi/full/10.1111/1751-7915.13876"
  },
  {
    title: "Comprehensive genomic analysis of Bacillus subtilis 9407 reveals its biocontrol potential against bacterial fruit blotchs",
    authors: "Xiaofei Gu, Qingchao Zeng, Yu Wang, Jishun Li, Yu Zhao, Yan Li, Qi Wang",
    year: 2021,
    topic: ["Molecular Biology", "Plant Microecology"],
    img: "https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs42483-021-00081-2/MediaObjects/42483_2021_81_Fig6_HTML.png?as=webp",
    link: "https://link.springer.com/article/10.1186/s42483-021-00081-2"
  },
  {
    title: "Interaction of viral pathogen with porin channels on the outer membrane of insect bacterial symbionts mediates their joint transovarial transmission",
    authors: "Wei Wu, Lingzhi Huang, Qianzhuo Mao, Jing Wei, Jiajia Li, Yu Zhao, Qian Zhang, Dongsheng Jia, Taiyun Wei",
    year: 2019,
    topic: ["Molecular Biology", "Plant Microecology"],
    img: "https://royalsocietypublishing.org/cms/asset/e5ffd8c7-8cb4-474c-a121-54e0944a0452/rstb20180320f01.jpg",
    link: "https://royalsocietypublishing.org/doi/full/10.1098/rstb.2018.0320"
  },
  // 添加更多数据...
];

let selectedYear = "all";
let selectedTopic = "all";

const renderPapers = () => {
  const list = document.getElementById("papers-list");
  list.innerHTML = "";

  const filtered = papers
    .filter(p => selectedYear === "all" || p.year == selectedYear)
    .filter(p => selectedTopic === "all" || p.topic.includes(selectedTopic))
    .sort((a, b) => b.year - a.year);

  filtered.forEach(p => {
    const div = document.createElement("div");
    div.className = "paper";
    div.innerHTML = `
      <img src="${p.img}" alt="preview">
      <div class="paper-info">
        <strong>${p.title}</strong>
        <p><em>${p.authors}</em></p>
        <p>${p.year} | <a href="${p.link}">PDF</a></p>
      </div>
    `;
    list.appendChild(div);
  });
};

document.querySelectorAll("#year-filter .filter-tag").forEach(el =>
  el.addEventListener("click", () => {
    document.querySelectorAll("#year-filter .filter-tag").forEach(tag => tag.classList.remove("active"));
    el.classList.add("active");
    selectedYear = el.dataset.year;
    renderPapers();
  })
);

document.querySelectorAll("#topic-filter .filter-tag").forEach(el =>
  el.addEventListener("click", () => {
    document.querySelectorAll("#topic-filter .filter-tag").forEach(tag => tag.classList.remove("active"));
    el.classList.add("active");
    selectedTopic = el.dataset.topic;
    renderPapers();
  })
);

renderPapers();
