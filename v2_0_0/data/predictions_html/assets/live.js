// assets/live.js
(function () {
    function parseJST(dtStr) {
      // "2026-01-24T15:45:00+09:00" 形式を想定
      const d = new Date(dtStr);
      return isNaN(d.getTime()) ? null : d;
    }
  
    function update() {
      const now = new Date();
      const rows = document.querySelectorAll("tr[data-post-dt]");
  
      rows.forEach((tr) => {
        const dt = parseJST(tr.getAttribute("data-post-dt"));
        if (!dt) return;
  
        // clear time classes
        tr.classList.remove("time-soon", "time-verysoon", "time-passed", "time-live");
  
        const diffMin = (dt.getTime() - now.getTime()) / 60000;
  
        // 発走後
        if (diffMin < -2) {
          tr.classList.add("time-passed");
          return;
        }
  
        // 発走直前〜発走直後を "live" 扱い
        if (diffMin <= 2 && diffMin >= -2) {
          tr.classList.add("time-live");
          return;
        }
  
        // 10分以内
        if (diffMin <= 10) {
          tr.classList.add("time-verysoon");
          return;
        }
  
        // 60分以内
        if (diffMin <= 60) {
          tr.classList.add("time-soon");
          return;
        }
      });
    }
  
    update();
    setInterval(update, 60 * 1000); // 1分ごと
  })();
  