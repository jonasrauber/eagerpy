module.exports = {
  base: '/eagerpy/',
  title: 'EagerPy',
  description: 'A unified API for PyTorch, TensorFlow, JAX and NumPy',
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'API', link: '/api/' },
      { text: 'GitHub', link: 'https://github.com/jonasrauber/eagerpy' }
    ],
    sidebar: [
      {
        title: 'Guide',
        collapsable: false,
        children: [
          '/guide/',
          '/guide/getting-started',
          '/guide/examples',
        ],
      },
      {
        title: 'API',
        collapsable: false,
        children: [
          '/api/',
          ['/api/tensor', 'Tensor'],
          '/api/lib',
          '/api/norms',
          '/api/types',
        ],
      },
    ],
  },
}
