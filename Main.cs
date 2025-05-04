using System;
using Fiourp;
using AI;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

namespace CartPole
{
    public class Main : Game
    {
        public static Main game1;
        private GraphicsDeviceManager _graphics;
        private SpriteBatch _spriteBatch;

        public static int episode;
        Env2 env;
        
        public Main()
        {
            _graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = true;
            game1 = this;

            _graphics.SynchronizeWithVerticalRetrace = false;
            IsFixedTimeStep = false;
        }

        protected override void Initialize()
        {
            // TODO: Add your initialization logic here

            Engine.Initialize(_graphics, Content, 1280, 720, new RenderTarget2D(GraphicsDevice, 1280, 720), "");
            
            base.Initialize();
        }

        protected override void LoadContent()
        {
            _spriteBatch = new SpriteBatch(GraphicsDevice);
            Drawing.Init(_spriteBatch, Content.Load<SpriteFont>("font"));

            Engine.CurrentMap = new Map();
            Engine.Cam = new Camera(Vector2.Zero, 0, 1);

            env = new Env2();
        }

        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed ||
                Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();
            
            Input.UpdateState();

            env.DoStep();

            Input.UpdateOldState();

            if (Input.GetKeyDown(Keys.Space))
            {
                _graphics.SynchronizeWithVerticalRetrace = !_graphics.SynchronizeWithVerticalRetrace;
                IsFixedTimeStep = !IsFixedTimeStep;
                _graphics.ApplyChanges();
            }
            
            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.Black);

            _spriteBatch.Begin();

            env.Render();
            
            Drawing.DebugString();
            Drawing.DebugPoint(4, 1);
            Drawing.DebugEvents();
            
            _spriteBatch.End();

            base.Draw(gameTime);
        }

        public void Stop()
        {
            Exit();
        }
    }
}