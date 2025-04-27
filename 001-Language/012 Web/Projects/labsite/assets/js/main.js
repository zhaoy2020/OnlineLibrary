// assets/js/main.js

// 加载导航栏
fetch('components/nav.html')
  .then(response => response.text())
  .then(data => {
    document.getElementById('navbar').innerHTML = data;
  });

// 加载页脚
fetch('components/footer.html')
  .then(response => response.text())
  .then(data => {
    document.getElementById('footer').innerHTML = data;
  });
