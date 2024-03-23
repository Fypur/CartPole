using System;
using Fiourp;
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
        Env3 env;
        
        public Main()
        {
            _graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = true;
            game1 = this;

            NN2 nn = new(new int[]{ 3, 3, 3 }, 0.01f);
            nn.Biases = new float[][]{
                null,
                new float[] { 2, 2, 2 },
                new float[] { 2, 2, 2 },
            };

            nn.Weights = new float[][][]{
                null,
                new float[][]{
                    new float[]{ 1,1,1 },
                    new float[]{ 1,1,1 },
                    new float[]{ 1,1,1 }
                },
                new float[][]{
                    new float[]{ 1,1,1 },
                    new float[]{ 1,1,1 },
                    new float[]{ 1,1,1 }
                }
            };
            nn.FeedForward(new float[] { 0, 0, 0 });

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

            env = new Env3();
        }

        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed ||
                Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();
            
            Input.UpdateState();

            env.Update();

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