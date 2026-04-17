import { PrismaClient } from "../src/generated/prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import bcrypt from "bcryptjs";

const prisma = new PrismaClient({
  adapter: new PrismaPg({ connectionString: process.env.DATABASE_URL }),
});

async function main() {
  console.log("🌱 Seeding database...");

  // ── Admin User ──
  const existing = await prisma.adminUser.count();
  if (existing === 0) {
    const hashed = await bcrypt.hash("admin123", 12);
    await prisma.adminUser.create({
      data: { email: "admin@devsuvam.dev", password: hashed, name: "Admin", role: "super_admin" },
    });
    console.log("✅ Created admin user (admin@devsuvam.dev / admin123)");
  }

  // ── Hero ──
  await prisma.heroContent.deleteMany();
  await prisma.heroContent.create({
    data: {
      name: "Jyotishankar",
      title: "I'm Jyotishankar, a Backend focused Full-Stack Developer based in India",
      description: "I'm passionate about creating meaningful digital experiences, with a strong focus on backend development and building reliable, scalable systems.",
      resumeUrl: "https://res.cloudinary.com/djby1yfko/image/upload/v1768676198/Jyotishankars_resume_twuz3f.pdf",
      images: [
        "https://res.cloudinary.com/djby1yfko/image/upload/v1768664109/IMG_20251208_165303878_1_bg5nr2.jpg",
        "https://res.cloudinary.com/djby1yfko/image/upload/v1768664101/IMG_20251208_165343292_jlvexy.jpg",
        "https://res.cloudinary.com/djby1yfko/image/upload/v1768664054/20260112_134703-IMG_STYLE_sit7ue.jpg",
      ],
      techBadges: ["Node JS", "PostgreSQL", "Docker", "Redis"],
      ctaPrimary: "Read the blog",
      ctaPrimaryUrl: "/blogs",
      ctaSecondary: "Resume",
      ctaSecondaryUrl: "https://res.cloudinary.com/djby1yfko/image/upload/v1768676198/Jyotishankars_resume_twuz3f.pdf",
    },
  });
  console.log("✅ Hero content seeded");

  // ── About ──
  await prisma.aboutContent.deleteMany();
  await prisma.aboutContent.create({
    data: {
      name: "Jyotishankar Patra",
      title: "Full Stack Software Developer (Backend-Focused)",
      bio: "I'm a full stack software developer with more focus on backend development. I build scalable and reliable backend systems and also have skills in frontend development. I'm good at working with databases, creating APIs, and using modern tools to make efficient software. My experience comes from building full-stack products using modern tools like Next.js, PostgreSQL, Docker, and cloud infrastructure.",
      avatarUrl: "https://res.cloudinary.com/djby1yfko/image/upload/v1768664109/IMG_20251208_165303878_1_bg5nr2.jpg",
      email: "jyotipatra.subham@gmail.com",
      phone: "+91-9861250893",
      linkedin: "https://linkedin.com/in/jyotishankar-patra",
      github: "https://github.com/jyotishankar04",
      strengths: JSON.stringify([
        { icon: "Code2", title: "Full-Stack Development", desc: "Built multiple production-grade applications" },
        { icon: "Server", title: "Backend Architecture", desc: "System design & microservices expertise" },
        { icon: "Database", title: "Database Management", desc: "PostgreSQL, MongoDB, Redis, Prisma ORM" },
        { icon: "CheckCircle2", title: "DevOps & Deployment", desc: "Docker, AWS, CI/CD, Cloudflare" },
      ]),
    },
  });
  console.log("✅ About content seeded");

  // ── Projects ──
  await prisma.project.deleteMany();
  await prisma.project.createMany({
    data: [
      {
        slug: "quickbrain-ai", title: "QuickBrain AI",
        description: "AI-Powered Quiz Generator platform using LangChain, Gemini AI, and Pinecone for vector-based question retrieval.",
        tags: ["TypeScript", "LangChain", "Pinecone", "Gemini AI", "Next.js", "Tailwind CSS"],
        image: "https://res.cloudinary.com/djby1yfko/image/upload/v1746005123/Screenshot_from_2025-03-28_20-20-43_xnnmqe.png",
        liveUrl: "https://quickbrainai.netlify.app", githubUrl: "https://github.com/jyotishankar04/quickBrainAI-frontend",
        featured: true, sortOrder: 1, status: "published",
        content: "## Overview\n\nQuickBrain AI is an AI-powered quiz generation platform that leverages LangChain for orchestration, Gemini AI for content generation, and Pinecone for vector-based question retrieval.\n\n## Key Features\n\n- **AI-Powered Quiz Generation** — Generate quizzes on any topic\n- **Vector Search** — Semantic search for finding relevant questions\n- **Real-time Feedback** — Instant scoring and explanations\n- **Responsive Design** — Built with Next.js and Tailwind CSS",
      },
      {
        slug: "quizzify", title: "Quizzify",
        description: "Interactive Quiz Platform with AI question generation (Gemini API) and real-time feedback system.",
        tags: ["React", "Node.js", "PostgreSQL", "AI", "JWT", "Tailwind CSS"],
        image: "https://res.cloudinary.com/djby1yfko/image/upload/v1741372603/Screenshot_from_2025-03-08_00-05-22_tsohbh.png",
        githubUrl: "https://github.com/jyotishankar04/Quizzify",
        featured: true, sortOrder: 2, status: "published",
        content: "## Overview\n\nQuizzify is an interactive quiz platform with AI-powered question generation using the Gemini API and real-time feedback.\n\n## Key Features\n\n- **AI Question Generation** — Dynamic quiz creation via Gemini API\n- **Real-time Scoring** — Instant feedback on answers\n- **User Authentication** — JWT-based auth system\n- **PostgreSQL Backend** — Reliable data storage with Prisma ORM",
      },
      {
        slug: "watch-ecommerce", title: "Watch E-commerce",
        description: "End-to-end e-commerce platform with Next.js, Express, and PostgreSQL with PhonePe payment integration.",
        tags: ["Next.js", "Express", "PostgreSQL", "E-commerce", "Payment", "Tailwind CSS"],
        image: "https://res.cloudinary.com/djby1yfko/image/upload/v1741372858/Screenshot_from_2025-03-08_00-10-46_yauyzv.png",
        githubUrl: "https://github.com/jyotishankar04/JustWatches",
        sortOrder: 3, status: "published",
        content: "## Overview\n\nA full-featured e-commerce platform for luxury watches built with Next.js and Express.\n\n## Key Features\n\n- **Product Catalog** — Browse and search watches by category\n- **Shopping Cart** — Add, remove, and manage items\n- **PhonePe Integration** — Seamless UPI payments",
      },
      {
        slug: "nexgpt", title: "NexGPT",
        description: "Chat bot project using Google Gemini API with Cloudflare integration.",
        tags: ["Next.js", "AI", "Gemini", "Cloudflare"],
        image: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=800&h=600&fit=crop",
        githubUrl: "https://github.com/jyotishankar04/s-gpt",
        sortOrder: 4, status: "published",
        content: "## Overview\n\nNexGPT is a chatbot project using Google Gemini API with Cloudflare integration.\n\n## Key Features\n\n- **AI Chat** — Conversational AI powered by Gemini\n- **Cloudflare Integration** — Edge-optimized performance",
      },
      {
        slug: "expense-tracker", title: "Expense Mate",
        description: "A simple expense tracker app using nextjs and postgres with Cloudflare integration",
        tags: ["Next.js", "PostgreSQL", "Finance", "Cloudflare"],
        image: "https://res.cloudinary.com/djby1yfko/image/upload/v1741373061/Screenshot_from_2025-03-08_00-14-08_y7xft0.png",
        githubUrl: "https://github.com/jyotishankar04/expense-tracker",
        sortOrder: 5, status: "published",
        content: "## Overview\n\nExpense Mate is a simple yet powerful expense tracking application built with Next.js and PostgreSQL.\n\n## Key Features\n\n- **Track Expenses** — Log and categorize daily expenses\n- **Visual Reports** — Charts and summaries of spending patterns",
      },
    ],
  });
  console.log("✅ 5 projects seeded");

  // ── Experience ──
  await prisma.experience.deleteMany();
  await prisma.experience.create({
    data: {
      company: "Ansmake Technology",
      role: "Full Stack Developer Intern",
      period: "Jun 2025 - Aug 2025",
      works: [
        "Developed and optimized RESTful APIs using FASTAPI, improving response times by 30%",
        "Implemented frontend components with Next.js, Tanstack Query, and Radix UI, enhancing user experience",
        "Collaborated with cross-functional teams to deliver SaaS product features on schedule",
      ],
      skills: ["FastAPI", "Next.js", "React", "TypeScript", "PostgreSQL", "Tailwind CSS", "Python", "Git"],
      sortOrder: 1,
    },
  });
  console.log("✅ Experience seeded");

  // ── Education ──
  await prisma.education.deleteMany();
  await prisma.education.createMany({
    data: [
      {
        degree: "Master of Computer Applications (MCA)",
        institution: "Kalinga Institute of Industrial Technology, Bhubaneswar",
        institutionUrl: "https://www.kiit.ac.in/",
        duration: "Aug 2025 - Present",
        isCurrent: true, sortOrder: 1,
      },
      {
        degree: "Bachelor of Science in Computer Science (BSCS)",
        institution: "Fakir Mohan Autonomous College, Balasore",
        institutionUrl: "https://www.fmcollege.nic.in/",
        duration: "Aug 2022 - May 2025",
        gpa: "7.03", sortOrder: 2,
      },
      {
        degree: "12th (PCM)",
        institution: "Ramarani Institute of engineering and technology, Balasore",
        duration: "Aug 2020 - May 2022",
        gpa: "79%", sortOrder: 3,
      },
    ],
  });
  console.log("✅ Education seeded");

  // ── Skills ──
  await prisma.skill.deleteMany();
  await prisma.skillCategory.deleteMany();

  const skillsData = [
    { name: "Languages", icon: "Code", skills: [
      { name: "TypeScript", icon: "SiTypescript" }, { name: "JavaScript", icon: "SiJavascript" },
      { name: "Python", icon: "SiPython" }, { name: "C++", icon: "SiCplusplus" },
    ]},
    { name: "Frontend", icon: "Code", skills: [
      { name: "React", icon: "SiReact" }, { name: "Next.js", icon: "SiNextdotjs" },
      { name: "Tailwind", icon: "SiTailwindcss" }, { name: "Redux", icon: "SiRedux" },
    ]},
    { name: "Backend", icon: "Server", skills: [
      { name: "Node.js", icon: "SiNodedotjs" }, { name: "Express", icon: "SiExpress" },
      { name: "FastAPI", icon: "SiFastapi" }, { name: "GraphQL", icon: "SiGraphql" },
      { name: "Socket.io", icon: "SiSocketdotio" },
    ]},
    { name: "Databases", icon: "Database", skills: [
      { name: "PostgreSQL", icon: "SiPostgresql" }, { name: "MongoDB", icon: "SiMongodb" },
      { name: "Redis", icon: "SiRedis" }, { name: "Prisma", icon: "SiPrisma" },
    ]},
    { name: "DevOps & Tools", icon: "Cloud", skills: [
      { name: "Docker", icon: "SiDocker" }, { name: "AWS", icon: "SiAwsamplify" },
      { name: "Nginx", icon: "SiNginx" }, { name: "Git", icon: "SiGit" },
      { name: "GitHub", icon: "SiGithub" }, { name: "Linux", icon: "SiLinux" },
      { name: "Postman", icon: "SiPostman" },
    ]},
    { name: "AI Engineering", icon: "Brain", skills: [
      { name: "LangChain", icon: "SiLangchain" },
    ]},
  ];

  let skillOrder = 1;
  for (const cat of skillsData) {
    const category = await prisma.skillCategory.create({
      data: { name: cat.name, icon: cat.icon, sortOrder: skillOrder++ },
    });
    let sOrder = 1;
    for (const skill of cat.skills) {
      await prisma.skill.create({
        data: { name: skill.name, icon: skill.icon, categoryId: category.id, sortOrder: sOrder++ },
      });
    }
  }
  console.log("✅ Skills & categories seeded");

  // ── Social Links ──
  await prisma.socialLink.deleteMany();
  await prisma.socialLink.createMany({
    data: [
      { name: "LinkedIn", url: "https://linkedin.com/in/jyotishankar-patra", username: "jyotishankar-patra", icon: "Linkedin", platform: "linkedin", sortOrder: 1 },
      { name: "Instagram", url: "https://instagram.com/dev.suvam", username: "@dev.suvam", icon: "Instagram", platform: "instagram", sortOrder: 2 },
      { name: "Email", url: "mailto:jyotipatra.subham@gmail.com", username: "jyotipatra.subham@gmail.com", icon: "Mail", platform: "email", sortOrder: 3 },
      { name: "GitHub", url: "https://github.com/jyotishankar04", username: "jyotishankar04", icon: "Github", platform: "github", sortOrder: 4 },
      { name: "Twitter", url: "https://twitter.com/devsuvam1", username: "devsuvam1", icon: "Twitter", platform: "twitter", sortOrder: 5 },
    ],
  });
  console.log("✅ Social links seeded");

  // ── Gallery ──
  await prisma.galleryItem.deleteMany();
  await prisma.galleryItem.createMany({
    data: [
      { image: "https://res.cloudinary.com/djby1yfko/image/upload/v1768675934/Snapchat-136706696_mwiu1y.jpg", caption: "Locking in. Quiet nights, loud goals.", mood: "Focused", date: "Jan 2026", featured: true, sortOrder: 1 },
      { image: "https://res.cloudinary.com/djby1yfko/image/upload/v1768675716/Snapchat-1197930273_bzrsrn.jpg", caption: "Sunlight fixes more than we think.", mood: "Calm", date: "Dec 2025", sortOrder: 2 },
      { image: "https://res.cloudinary.com/djby1yfko/image/upload/v1768664101/IMG_20251208_165343292_jlvexy.jpg", caption: "Head down. World muted.", mood: "Locked in", date: "Nov 2025", sortOrder: 3 },
      { image: "https://res.cloudinary.com/djby1yfko/image/upload/v1768676100/IMG_0032_zv1pf9.jpg", caption: "Progress doesn't need permission.", mood: "Inspired", date: "Oct 2025", sortOrder: 4 },
      { image: "https://res.cloudinary.com/djby1yfko/image/upload/v1768675939/Snapchat-1287290365_g0it96.jpg", caption: "Just showing up matters.", mood: "Reflective", date: "Sep 2025", sortOrder: 5 },
    ],
  });
  console.log("✅ Gallery seeded");

  // ── Navigation ──
  await prisma.navigationItem.deleteMany();
  await prisma.navigationItem.createMany({
    data: [
      { label: "Home", path: "/", icon: "Home", section: "header", sortOrder: 1 },
      { label: "Blogs", path: "/blogs", icon: "BookOpen", section: "header", sortOrder: 2 },
      { label: "About", path: "/about", icon: "User", section: "header", sortOrder: 3 },
      { label: "Experience", path: "/experience", icon: "Briefcase", section: "header", sortOrder: 4 },
      { label: "Projects", path: "/projects", icon: "Code", section: "header", sortOrder: 5 },
      { label: "Chat", path: "/chat", icon: "Mail", section: "header", sortOrder: 6 },
    ],
  });
  console.log("✅ Navigation seeded");

  console.log("\n🎉 Seed complete!");
  console.log("   Admin login: admin@devsuvam.dev / admin123");
}

main()
  .catch((e) => { console.error(e); process.exit(1); })
  .finally(() => prisma.$disconnect());
