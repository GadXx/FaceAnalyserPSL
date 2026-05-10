import { createRouter, createWebHistory } from "vue-router";

import AnalyzeView from "../views/AnalyzeView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      name: "analyze",
      component: AnalyzeView,
    },
  ],
});

export default router;
