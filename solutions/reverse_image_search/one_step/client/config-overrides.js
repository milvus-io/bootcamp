module.exports = {
  webpack: function (config, env) {
    config.devtool = false;
    config.plugins = config.plugins.filter((p) => p.constructor.name !== "ManifestPlugin" && p.constructor.name !== "GenerateSW");

    // config.optimization.runtimeChunk = false;
    // config.optimization.splitChunks = {
    //   cacheGroups: {
    //     default: false,
    //   },
    // };


    // console.log(config.plugins.map((p) => p.constructor.name));
    // config.module.rules.map((rule) => {
    //   console.log(rule.use);
    //   console.log();
    // });

    // config.output.filename = "static/js/app.js";

    return config;
  },
};
