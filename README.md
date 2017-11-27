# 关于此 blog 

- Blog 使用 Jekyll + Lanyon 构建。

- 字体构建参考了[这个帖子](http://longqian.me/2017/02/12/jekyll-support-chinese/).

- 修改了 `_includes/head.html`：
    - 添加了 `LaTeX` 支持；
    - 添加了 `font.css`, `font-chinese.css`, `syntax.css`

- `_layouts` 下修改了 `page.html` 和 `post.html` 以支持中文配置字体:

  ```
  {% if page.lang %}
  <div class="page" lang="{{page.lang}}">
  {% else %}
  <div class="page">
  {% endif %}
  ```
 
- 修改了 `index.html` 以应用 home page 的中文修正。 


- `public/css` 目录下添加了 `fonts.css`, `fonts-chinese.css`, `syntax.css` 以及中文字体 `fonts-local`。

- 在 `public/css/poole.css` 中添加了一下行以支持中文字体：

  ```css
  h1, h2, h1 a, h2 a:lang(zh) {
  font-family: "PT Sans", "cwTeXMing", serif;
  }
  h3, h4, h5, h6, p, table, img, a, ul, li:lang(zh) {
  font-family: "PT Sans", "Noto Sans SC", "Microsoft YaHei", sans-serif;
  }
  blockquote p:lang(zh){
  font-family: "PT Sans", "STKaiti", "Kaiti", "cwTeXKai", "Microsoft YaHei", sans-serif;
  }
  ```

- 关于代码块高亮，使用`rougify style monokai.sublime > assets/css/syntax.css` 产生高亮格式。需要注意的是，如果使用深色背景(比如本 blog 的 `monokai` 主题)，需要在产生的 `syntax.css` 里添加下面一行。

  ```
  pre[class='highlight'] {background-color:#000000;}
  ``` 
  详见[这个帖子](https://oncemore2020.github.io/blog/upgrade-jekyll/)。



- 关于home里帖子链接不正确，修改 `index.html` 的 `{{site.baseurl}}/{{post.url}}` 为

```html
{{site.baseurl}}{{post.url}}
```

